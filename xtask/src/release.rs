use crate::{
    codegen, is_release_tag,
    not_bash::{date_iso, fs2, run},
    project_root, Mode, Result,
};

pub struct ReleaseCmd {
    pub dry_run: bool,
}

impl ReleaseCmd {
    pub fn run(self) -> Result<()> {
        if !self.dry_run {
            run!("git switch release")?;
            run!("git fetch upstream --tags --force")?;
            run!("git reset --hard tags/nightly")?;
            run!("git push")?;
        }
        codegen::generate_assists_docs(Mode::Overwrite)?;
        codegen::generate_feature_docs(Mode::Overwrite)?;

        let website_root = project_root().join("../rust-analyzer.github.io");
        let changelog_dir = website_root.join("./thisweek/_posts");

        let today = date_iso()?;
        let commit = run!("git rev-parse HEAD")?;
        let changelog_n = fs2::read_dir(changelog_dir.as_path())?.count();

        let contents = format!(
            "\
= Changelog #{}
:sectanchors:
:page-layout: post

Commit: commit:{}[] +
Release: release:{}[]

== Sponsors

== New Features

* pr:[] .

== Fixes

== Internal Improvements
",
            changelog_n, commit, today
        );

        let path = changelog_dir.join(format!("{}-changelog-{}.adoc", today, changelog_n));
        fs2::write(&path, &contents)?;

        for &adoc in ["manual.adoc", "generated_features.adoc", "generated_assists.adoc"].iter() {
            let src = project_root().join("./docs/user/").join(adoc);
            let dst = website_root.join(adoc);
            fs2::copy(src, dst)?;
        }

        let tags = run!("git tag --list"; echo = false)?;
        let prev_tag = tags.lines().filter(|line| is_release_tag(line)).last().unwrap();

        let git_log = run!("git log {}..HEAD --merges --reverse", prev_tag; echo = false)?;
        let git_log_dst = website_root.join("git.log");
        fs2::write(git_log_dst, &git_log)?;

        Ok(())
    }
}
