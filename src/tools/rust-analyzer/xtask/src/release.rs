mod changelog;

use xshell::{Shell, cmd};

use crate::{date_iso, flags, is_release_tag, project_root};

impl flags::Release {
    pub(crate) fn run(self, sh: &Shell) -> anyhow::Result<()> {
        if !self.dry_run {
            cmd!(sh, "git switch release").run()?;
            cmd!(sh, "git fetch upstream --tags --force").run()?;
            cmd!(sh, "git reset --hard tags/nightly").run()?;
            // The `release` branch sometimes has a couple of cherry-picked
            // commits for patch releases. If that's the case, just overwrite
            // it. As we are setting `release` branch to an up-to-date `nightly`
            // tag, this shouldn't be problematic in general.
            //
            // Note that, as we tag releases, we don't worry about "losing"
            // commits -- they'll be kept alive by the tag. More generally, we
            // don't care about historic releases all that much, it's fine even
            // to delete old tags.
            cmd!(sh, "git push --force").run()?;
        }

        let website_root = project_root().join("../rust-analyzer.github.io");
        {
            let _dir = sh.push_dir(&website_root);
            cmd!(sh, "git switch src").run()?;
            cmd!(sh, "git pull").run()?;
        }
        let changelog_dir = website_root.join("./thisweek/_posts");

        let today = date_iso(sh)?;
        let commit = cmd!(sh, "git rev-parse HEAD").read()?;
        let changelog_n = sh
            .read_dir(changelog_dir.as_path())?
            .into_iter()
            .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
            .filter_map(|s| s.splitn(5, '-').last().map(|n| n.replace('-', ".")))
            .filter_map(|s| s.parse::<f32>().ok())
            .map(|n| 1 + n.floor() as usize)
            .max()
            .unwrap_or_default();

        let tags = cmd!(sh, "git tag --list").read()?;
        let prev_tag = tags.lines().filter(|line| is_release_tag(line)).next_back().unwrap();

        let contents = changelog::get_changelog(sh, changelog_n, &commit, prev_tag, &today)?;
        let path = changelog_dir.join(format!("{today}-changelog-{changelog_n}.adoc"));
        sh.write_file(path, contents)?;

        Ok(())
    }
}
