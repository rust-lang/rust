use std::path::{Path, PathBuf};

use build_helper::t;

use crate::builder::Builder;

#[derive(Copy, Clone)]
pub(crate) enum OverlayKind {
    Rust,
    LLVM,
}

impl OverlayKind {
    fn included_files(&self) -> &[&str] {
        match self {
            OverlayKind::Rust => &["COPYRIGHT", "LICENSE-APACHE", "LICENSE-MIT", "README.md"],
            OverlayKind::LLVM => {
                &["src/llvm-project/llvm/LICENSE.TXT", "src/llvm-project/llvm/README.txt"]
            }
        }
    }
}

pub(crate) struct Tarball<'a> {
    builder: &'a Builder<'a>,

    pkgname: String,
    component: String,
    target: String,
    overlay: OverlayKind,

    temp_dir: PathBuf,
    image_dir: PathBuf,
    overlay_dir: PathBuf,
    work_dir: PathBuf,
}

impl<'a> Tarball<'a> {
    pub(crate) fn new(builder: &'a Builder<'a>, component: &str, target: &str) -> Self {
        let pkgname = crate::dist::pkgname(builder, component);

        let temp_dir = builder.out.join("tmp").join("tarball").join(component);
        let _ = std::fs::remove_dir_all(&temp_dir);

        let image_dir = temp_dir.join("image");
        let overlay_dir = temp_dir.join("overlay");
        let work_dir = temp_dir.join("work");

        Self {
            builder,

            pkgname,
            component: component.into(),
            target: target.into(),
            overlay: OverlayKind::Rust,

            temp_dir,
            image_dir,
            overlay_dir,
            work_dir,
        }
    }

    pub(crate) fn set_overlay(&mut self, overlay: OverlayKind) {
        self.overlay = overlay;
    }

    pub(crate) fn image_dir(&self) -> &Path {
        t!(std::fs::create_dir_all(&self.image_dir));
        &self.image_dir
    }

    pub(crate) fn add_file(&self, src: impl AsRef<Path>, destdir: impl AsRef<Path>, perms: u32) {
        // create_dir_all fails to create `foo/bar/.`, so when the destination is "." this simply
        // uses the base directory as the destination directory.
        let destdir = if destdir.as_ref() == Path::new(".") {
            self.image_dir.clone()
        } else {
            self.image_dir.join(destdir.as_ref())
        };

        t!(std::fs::create_dir_all(&destdir));
        self.builder.install(src.as_ref(), &destdir, perms);
    }

    pub(crate) fn add_dir(&self, src: impl AsRef<Path>, destdir: impl AsRef<Path>) {
        t!(std::fs::create_dir_all(destdir.as_ref()));
        self.builder.cp_r(
            src.as_ref(),
            &self.image_dir.join(destdir.as_ref()).join(src.as_ref().file_name().unwrap()),
        );
    }

    pub(crate) fn generate(self) -> PathBuf {
        t!(std::fs::create_dir_all(&self.overlay_dir));
        self.builder.create(&self.overlay_dir.join("version"), &self.builder.rust_version());
        for file in self.overlay.included_files() {
            self.builder.install(&self.builder.src.join(file), &self.overlay_dir, 0o644);
        }

        let distdir = crate::dist::distdir(self.builder);
        let mut cmd = self.builder.tool_cmd(crate::tool::Tool::RustInstaller);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg(format!("--success-message={} installed.", self.component))
            .arg("--image-dir")
            .arg(self.image_dir)
            .arg("--work-dir")
            .arg(self.work_dir)
            .arg("--output-dir")
            .arg(&distdir)
            .arg("--non-installed-overlay")
            .arg(self.overlay_dir)
            .arg(format!("--package-name={}-{}", self.pkgname, self.target))
            .arg("--legacy-manifest-dirs=rustlib,cargo")
            .arg(format!("--component-name={}", self.component));
        self.builder.run(&mut cmd);

        t!(std::fs::remove_dir_all(&self.temp_dir));

        distdir.join(format!("{}-{}.tar.gz", self.pkgname, self.target))
    }
}
