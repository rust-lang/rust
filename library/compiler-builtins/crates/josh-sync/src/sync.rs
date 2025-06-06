use std::net::{SocketAddr, TcpStream};
use std::process::{Command, Stdio, exit};
use std::time::Duration;
use std::{env, fs, process, thread};

const JOSH_PORT: u16 = 42042;
const DEFAULT_PR_BRANCH: &str = "update-builtins";

pub struct GitSync {
    upstream_repo: String,
    upstream_ref: String,
    upstream_url: String,
    josh_filter: String,
    josh_url_base: String,
}

/// This code was adapted from the miri repository, via the rustc-dev-guide
/// (<https://github.com/rust-lang/rustc-dev-guide/tree/c51adbd12d/josh-sync>)
impl GitSync {
    pub fn from_current_dir() -> Self {
        let upstream_repo =
            env::var("UPSTREAM_ORG").unwrap_or_else(|_| "rust-lang".to_owned()) + "/rust";

        Self {
            upstream_url: format!("https://github.com/{upstream_repo}"),
            upstream_repo,
            upstream_ref: env::var("UPSTREAM_REF").unwrap_or_else(|_| "HEAD".to_owned()),
            josh_filter: ":/library/compiler-builtins".to_owned(),
            josh_url_base: format!("http://localhost:{JOSH_PORT}"),
        }
    }

    /// Pull from rust-lang/rust to compiler-builtins.
    pub fn rustc_pull(&self, commit: Option<String>) {
        let Self {
            upstream_ref,
            upstream_url,
            upstream_repo,
            ..
        } = self;

        let new_upstream_base = commit.unwrap_or_else(|| {
            let out = check_output(["git", "ls-remote", upstream_url, upstream_ref]);
            out.split_whitespace()
                .next()
                .unwrap_or_else(|| panic!("could not split output: '{out}'"))
                .to_owned()
        });

        ensure_clean();

        // Make sure josh is running.
        let _josh = Josh::start();
        let josh_url_filtered = self.josh_url(
            &self.upstream_repo,
            Some(&new_upstream_base),
            Some(&self.josh_filter),
        );

        let previous_upstream_base = fs::read_to_string("rust-version")
            .expect("failed to read `rust-version`")
            .trim()
            .to_string();
        assert_ne!(previous_upstream_base, new_upstream_base, "nothing to pull");

        let orig_head = check_output(["git", "rev-parse", "HEAD"]);
        println!("original upstream base: {previous_upstream_base}");
        println!("new upstream base: {new_upstream_base}");
        println!("original HEAD: {orig_head}");

        // Fetch the latest upstream HEAD so we can get a summary. Use the Josh URL for caching.
        run([
            "git",
            "fetch",
            &self.josh_url(&self.upstream_repo, Some(&new_upstream_base), Some(":/")),
            &new_upstream_base,
            "--depth=1",
        ]);
        let new_summary = check_output(["git", "log", "-1", "--format=%h %s", &new_upstream_base]);

        // Update rust-version file. As a separate commit, since making it part of
        // the merge has confused the heck out of josh in the past.
        // We pass `--no-verify` to avoid running git hooks.
        // We do this before the merge so that if there are merge conflicts, we have
        // the right rust-version file while resolving them.
        fs::write("rust-version", format!("{new_upstream_base}\n"))
            .expect("failed to write rust-version");

        let prep_message = format!(
            "Update the upstream Rust version\n\n\
            To prepare for merging from {upstream_repo}, set the version file to:\n\n    \
            {new_summary}\n\
            ",
        );
        run([
            "git",
            "commit",
            "rust-version",
            "--no-verify",
            "-m",
            &prep_message,
        ]);

        // Fetch given rustc commit.
        run(["git", "fetch", &josh_url_filtered]);
        let incoming_ref = check_output(["git", "rev-parse", "FETCH_HEAD"]);
        println!("incoming ref: {incoming_ref}");

        let merge_message = format!(
            "Merge ref '{upstream_head_short}{filter}' from {upstream_url}\n\n\
            Pull recent changes from {upstream_repo} via Josh.\n\n\
            Upstream ref: {new_upstream_base}\n\
            Filtered ref: {incoming_ref}\n\
            ",
            upstream_head_short = &new_upstream_base[..12],
            filter = self.josh_filter
        );

        // This should not add any new root commits. So count those before and after merging.
        let num_roots = || -> u32 {
            let out = check_output(["git", "rev-list", "HEAD", "--max-parents=0", "--count"]);
            out.trim()
                .parse::<u32>()
                .unwrap_or_else(|e| panic!("failed to parse `{out}`: {e}"))
        };
        let num_roots_before = num_roots();

        let pre_merge_sha = check_output(["git", "rev-parse", "HEAD"]);
        println!("pre-merge HEAD: {pre_merge_sha}");

        // Merge the fetched commit.
        run([
            "git",
            "merge",
            "FETCH_HEAD",
            "--no-verify",
            "--no-ff",
            "-m",
            &merge_message,
        ]);

        let current_sha = check_output(["git", "rev-parse", "HEAD"]);
        if current_sha == pre_merge_sha {
            run(["git", "reset", "--hard", &orig_head]);
            eprintln!(
                "No merge was performed, no changes to pull were found. \
                Rolled back the preparation commit."
            );
            exit(1);
        }

        // Check that the number of roots did not increase.
        assert_eq!(
            num_roots(),
            num_roots_before,
            "Josh created a new root commit. This is probably not the history you want."
        );
    }

    /// Construct an update to rust-lang/rust from compiler-builtins.
    pub fn rustc_push(&self, github_user: &str, branch: Option<&str>) {
        let Self {
            josh_filter,
            upstream_url,
            ..
        } = self;

        let branch = branch.unwrap_or(DEFAULT_PR_BRANCH);
        let josh_url = self.josh_url(&format!("{github_user}/rust"), None, Some(josh_filter));
        let user_upstream_url = format!("git@github.com:{github_user}/rust.git");

        let Ok(rustc_git) = env::var("RUSTC_GIT") else {
            panic!("the RUSTC_GIT environment variable must be set to a rust-lang/rust checkout")
        };

        ensure_clean();
        let base = fs::read_to_string("rust-version")
            .expect("failed to read `rust-version`")
            .trim()
            .to_string();

        // Make sure josh is running.
        let _josh = Josh::start();

        // Prepare the branch. Pushing works much better if we use as base exactly
        // the commit that we pulled from last time, so we use the `rust-version`
        // file to find out which commit that would be.
        println!("Preparing {github_user}/rust (base: {base})...");

        if Command::new("git")
            .args(["-C", &rustc_git, "fetch", &user_upstream_url, branch])
            .output() // capture output
            .expect("could not run fetch")
            .status
            .success()
        {
            panic!(
                "The branch '{branch}' seems to already exist in '{user_upstream_url}'. \
                 Please delete it and try again."
            );
        }

        run(["git", "-C", &rustc_git, "fetch", upstream_url, &base]);

        run_cfg("git", |c| {
            c.args([
                "-C",
                &rustc_git,
                "push",
                &user_upstream_url,
                &format!("{base}:refs/heads/{branch}"),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null()) // silence the "create GitHub PR" message
        });
        println!("pushed PR branch");

        // Do the actual push.
        println!("Pushing changes...");
        run(["git", "push", &josh_url, &format!("HEAD:{branch}")]);
        println!();

        // Do a round-trip check to make sure the push worked as expected.
        run(["git", "fetch", &josh_url, branch]);

        let head = check_output(["git", "rev-parse", "HEAD"]);
        let fetch_head = check_output(["git", "rev-parse", "FETCH_HEAD"]);
        assert_eq!(
            head, fetch_head,
            "Josh created a non-roundtrip push! Do NOT merge this into rustc!\n\
             Expected {head}, got {fetch_head}."
        );
        println!(
            "Confirmed that the push round-trips back to compiler-builtins properly. Please \
            create a rustc PR:"
        );
        // Open PR with `subtree update` title to silence the `no-merges` triagebot check
        println!(
            "    {upstream_url}/compare/{github_user}:{branch}?quick_pull=1\
            &title=Update%20the%20%60compiler-builtins%60%20subtree\
            &body=Update%20the%20Josh%20subtree%20to%20https%3A%2F%2Fgithub.com%2Frust-lang%2F\
            compiler-builtins%2Fcommit%2F{head_short}.%0A%0Ar%3F%20%40ghost",
            head_short = &head[..12],
        );
    }

    /// Construct a url to the local Josh server with (optionally)
    fn josh_url(&self, repo: &str, rev: Option<&str>, filter: Option<&str>) -> String {
        format!(
            "{base}/{repo}.git{at}{rev}{filter}{filt_git}",
            base = self.josh_url_base,
            at = if rev.is_some() { "@" } else { "" },
            rev = rev.unwrap_or_default(),
            filter = filter.unwrap_or_default(),
            filt_git = if filter.is_some() { ".git" } else { "" }
        )
    }
}

/// Fail if there are files that need to be checked in.
fn ensure_clean() {
    let read = check_output(["git", "status", "--untracked-files=no", "--porcelain"]);
    assert!(
        read.is_empty(),
        "working directory must be clean before performing rustc pull"
    );
}

/* Helpers for running commands with logged invocations */

/// Run a command from an array, passing its output through.
fn run<'a, Args: AsRef<[&'a str]>>(l: Args) {
    let l = l.as_ref();
    run_cfg(l[0], |c| c.args(&l[1..]));
}

/// Run a command from an array, collecting its output.
fn check_output<'a, Args: AsRef<[&'a str]>>(l: Args) -> String {
    let l = l.as_ref();
    check_output_cfg(l[0], |c| c.args(&l[1..]))
}

/// [`run`] with configuration.
fn run_cfg(prog: &str, f: impl FnOnce(&mut Command) -> &mut Command) {
    // self.read(l.as_ref());
    check_output_cfg(prog, |c| f(c.stdout(Stdio::inherit())));
}

/// [`read`] with configuration. All shell helpers print the command and pass stderr.
fn check_output_cfg(prog: &str, f: impl FnOnce(&mut Command) -> &mut Command) -> String {
    let mut cmd = Command::new(prog);
    cmd.stderr(Stdio::inherit());
    f(&mut cmd);
    eprintln!("+ {cmd:?}");
    let out = cmd.output().expect("command failed");
    assert!(out.status.success());
    String::from_utf8(out.stdout.trim_ascii().to_vec()).expect("non-UTF8 output")
}

/// Create a wrapper that stops Josh on drop.
pub struct Josh(process::Child);

impl Josh {
    pub fn start() -> Self {
        // Determine cache directory.
        let user_dirs =
            directories::ProjectDirs::from("org", "rust-lang", "rustc-compiler-builtins-josh")
                .unwrap();
        let local_dir = user_dirs.cache_dir().to_owned();

        // Start josh, silencing its output.
        #[expect(clippy::zombie_processes, reason = "clippy can't handle the loop")]
        let josh = process::Command::new("josh-proxy")
            .arg("--local")
            .arg(local_dir)
            .args([
                "--remote=https://github.com",
                &format!("--port={JOSH_PORT}"),
                "--no-background",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to start josh-proxy, make sure it is installed");

        // Wait until the port is open. We try every 10ms until 1s passed.
        for _ in 0..100 {
            // This will generally fail immediately when the port is still closed.
            let addr = SocketAddr::from(([127, 0, 0, 1], JOSH_PORT));
            let josh_ready = TcpStream::connect_timeout(&addr, Duration::from_millis(1));

            if josh_ready.is_ok() {
                println!("josh up and running");
                return Josh(josh);
            }

            // Not ready yet.
            thread::sleep(Duration::from_millis(10));
        }
        panic!("Even after waiting for 1s, josh-proxy is still not available.")
    }
}

impl Drop for Josh {
    fn drop(&mut self) {
        if cfg!(unix) {
            // Try to gracefully shut it down.
            Command::new("kill")
                .args(["-s", "INT", &self.0.id().to_string()])
                .output()
                .expect("failed to SIGINT josh-proxy");
            // Sadly there is no "wait with timeout"... so we just give it some time to finish.
            thread::sleep(Duration::from_millis(100));
            // Now hopefully it is gone.
            if self
                .0
                .try_wait()
                .expect("failed to wait for josh-proxy")
                .is_some()
            {
                return;
            }
        }
        // If that didn't work (or we're not on Unix), kill it hard.
        eprintln!(
            "I have to kill josh-proxy the hard way, let's hope this does not \
            break anything."
        );
        self.0.kill().expect("failed to SIGKILL josh-proxy");
    }
}
