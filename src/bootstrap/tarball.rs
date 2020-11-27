use std::path::{Path, PathBuf};

use build_helper::t;

use crate::builder::Builder;

#[derive(Copy, Clone)]
pub(crate) enum OverlayKind {
    Rust,
    LLVM,
    Cargo,
    Clippy,
    Miri,
    Rustfmt,
    RLS,
}

impl OverlayKind {
    fn legal_and_readme(&self) -> &[&str] {
        match self {
            OverlayKind::Rust => &["COPYRIGHT", "LICENSE-APACHE", "LICENSE-MIT", "README.md"],
            OverlayKind::LLVM => {
                &["src/llvm-project/llvm/LICENSE.TXT", "src/llvm-project/llvm/README.txt"]
            }
            OverlayKind::Cargo => &[
                "src/tools/cargo/README.md",
                "src/tools/cargo/LICENSE-MIT",
                "src/tools/cargo/LICENSE-APACHE",
                "src/tools/cargo/LICENSE-THIRD-PARTY",
            ],
            OverlayKind::Clippy => &[
                "src/tools/clippy/README.md",
                "src/tools/clippy/LICENSE-APACHE",
                "src/tools/clippy/LICENSE-MIT",
            ],
            OverlayKind::Miri => &[
                "src/tools/miri/README.md",
                "src/tools/miri/LICENSE-APACHE",
                "src/tools/miri/LICENSE-MIT",
            ],
            OverlayKind::Rustfmt => &[
                "src/tools/rustfmt/README.md",
                "src/tools/rustfmt/LICENSE-APACHE",
                "src/tools/rustfmt/LICENSE-MIT",
            ],
            OverlayKind::RLS => &[
                "src/tools/rls/README.md",
                "src/tools/rls/LICENSE-APACHE",
                "src/tools/rls/LICENSE-MIT",
            ],
        }
    }

    fn version(&self, builder: &Builder<'_>) -> String {
        match self {
            OverlayKind::Rust => builder.rust_version(),
            OverlayKind::LLVM => builder.rust_version(),
            OverlayKind::Cargo => {
                builder.cargo_info.version(builder, &builder.release_num("cargo"))
            }
            OverlayKind::Clippy => {
                builder.clippy_info.version(builder, &builder.release_num("clippy"))
            }
            OverlayKind::Miri => builder.miri_info.version(builder, &builder.release_num("miri")),
            OverlayKind::Rustfmt => {
                builder.rustfmt_info.version(builder, &builder.release_num("rustfmt"))
            }
            OverlayKind::RLS => builder.rls_info.version(builder, &builder.release_num("rls")),
        }
    }
}

pub(crate) struct Tarball<'a> {
    builder: &'a Builder<'a>,

    pkgname: String,
    component: String,
    target: String,
    product_name: String,
    overlay: OverlayKind,

    temp_dir: PathBuf,
    image_dir: PathBuf,
    overlay_dir: PathBuf,
    work_dir: PathBuf,

    include_target_in_component_name: bool,
    is_preview: bool,
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
            product_name: "Rust".into(),
            overlay: OverlayKind::Rust,

            temp_dir,
            image_dir,
            overlay_dir,
            work_dir,

            include_target_in_component_name: false,
            is_preview: false,
        }
    }

    pub(crate) fn set_overlay(&mut self, overlay: OverlayKind) {
        self.overlay = overlay;
    }

    pub(crate) fn set_product_name(&mut self, name: &str) {
        self.product_name = name.into();
    }

    pub(crate) fn include_target_in_component_name(&mut self, include: bool) {
        self.include_target_in_component_name = include;
    }

    pub(crate) fn is_preview(&mut self, is: bool) {
        self.is_preview = is;
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

    pub(crate) fn add_renamed_file(
        &self,
        src: impl AsRef<Path>,
        destdir: impl AsRef<Path>,
        new_name: &str,
    ) {
        let destdir = self.image_dir.join(destdir.as_ref());
        t!(std::fs::create_dir_all(&destdir));
        self.builder.copy(src.as_ref(), &destdir.join(new_name));
    }

    pub(crate) fn add_legal_and_readme_to(&self, destdir: impl AsRef<Path>) {
        for file in self.overlay.legal_and_readme() {
            self.add_file(self.builder.src.join(file), destdir.as_ref(), 0o644);
        }
    }

    pub(crate) fn add_dir(&self, src: impl AsRef<Path>, dest: impl AsRef<Path>) {
        let dest = self.image_dir.join(dest.as_ref());

        t!(std::fs::create_dir_all(&dest));
        self.builder.cp_r(src.as_ref(), &dest);
    }

    pub(crate) fn generate(self) -> PathBuf {
        t!(std::fs::create_dir_all(&self.overlay_dir));
        self.builder.create(&self.overlay_dir.join("version"), &self.overlay.version(self.builder));
        if let Some(sha) = self.builder.rust_sha() {
            self.builder.create(&self.overlay_dir.join("git-commit-hash"), &sha);
        }
        for file in self.overlay.legal_and_readme() {
            self.builder.install(&self.builder.src.join(file), &self.overlay_dir, 0o644);
        }

        let mut cmd = self.builder.tool_cmd(crate::tool::Tool::RustInstaller);

        self.builder.info(&format!("Dist {} ({})", self.component, self.target));
        let _time = crate::util::timeit(self.builder);

        let mut component_name = self.component.clone();
        if self.is_preview {
            component_name.push_str("-preview");
        }
        if self.include_target_in_component_name {
            component_name.push('-');
            component_name.push_str(&self.target);
        }

        let distdir = crate::dist::distdir(self.builder);
        cmd.arg("generate")
            .arg(format!("--product-name={}", self.product_name))
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
            .arg(format!("--component-name={}", component_name));
        self.builder.run(&mut cmd);
        t!(std::fs::remove_dir_all(&self.temp_dir));

        distdir.join(format!("{}-{}.tar.gz", self.pkgname, self.target))
    }
}
