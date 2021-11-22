mod changelog;

use xshell::{cmd, pushd, read_dir, read_file, write_file};

use crate::{date_iso, flags, is_release_tag, project_root, Result};

impl flags::Release {
    pub(crate) fn run(self) -> Result<()> {
        if !self.dry_run {
            cmd!("git switch release").run()?;
            cmd!("git fetch upstream --tags --force").run()?;
            cmd!("git reset --hard tags/nightly").run()?;
            // The `release` branch sometimes has a couple of cherry-picked
            // commits for patch releases. If that's the case, just overwrite
            // it. As we are setting `release` branch to an up-to-date `nightly`
            // tag, this shouldn't be problematic in general.
            //
            // Note that, as we tag releases, we don't worry about "losing"
            // commits -- they'll be kept alive by the tag. More generally, we
            // don't care about historic releases all that much, it's fine even
            // to delete old tags.
            cmd!("git push --force").run()?;
        }

        // Generates bits of manual.adoc.
        cmd!("cargo test -p ide_assists -p ide_diagnostics -p rust-analyzer -- sourcegen_")
            .run()?;

        let website_root = project_root().join("../rust-analyzer.github.io");
        {
            let _dir = pushd(&website_root)?;
            cmd!("git switch src").run()?;
            cmd!("git pull").run()?;
        }
        let changelog_dir = website_root.join("./thisweek/_posts");

        let today = date_iso()?;
        let commit = cmd!("git rev-parse HEAD").read()?;
        let changelog_n = read_dir(changelog_dir.as_path())?.len();

        for adoc in [
            "manual.adoc",
            "generated_assists.adoc",
            "generated_config.adoc",
            "generated_diagnostic.adoc",
            "generated_features.adoc",
        ] {
            let src = project_root().join("./docs/user/").join(adoc);
            let dst = website_root.join(adoc);

            let contents = read_file(src)?;
            write_file(dst, contents)?;
        }

        let tags = cmd!("git tag --list").read()?;
        let prev_tag = tags.lines().filter(|line| is_release_tag(line)).last().unwrap();

        let contents = changelog::get_changelog(changelog_n, &commit, prev_tag, &today)?;
        let path = changelog_dir.join(format!("{}-changelog-{}.adoc", today, changelog_n));
        write_file(&path, &contents)?;

        Ok(())
    }
}

impl flags::Promote {
    pub(crate) fn run(self) -> Result<()> {
        let _dir = pushd("../rust-rust-analyzer")?;
        cmd!("git switch master").run()?;
        cmd!("git fetch upstream").run()?;
        cmd!("git reset --hard upstream/master").run()?;
        cmd!("git submodule update --recursive").run()?;

        let branch = format!("rust-analyzer-{}", date_iso()?);
        cmd!("git switch -c {branch}").run()?;
        {
            let _dir = pushd("src/tools/rust-analyzer")?;
            cmd!("git fetch origin").run()?;
            cmd!("git reset --hard origin/release").run()?;
        }
        cmd!("git add src/tools/rust-analyzer").run()?;
        cmd!("git commit -m':arrow_up: rust-analyzer'").run()?;
        if !self.dry_run {
            cmd!("git push -u origin {branch}").run()?;
            cmd!("xdg-open https://github.com/matklad/rust/pull/new/{branch}?body=r%3F%20%40ghost")
                .run()?;
        }
        Ok(())
    }
}
