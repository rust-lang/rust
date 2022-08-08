use crate::path::{Path, PathBuf};
use crate::process::Command;
use crate::sync::LazyLock;
use crate::sys_common::io::test::tmpdir;

static SCRIPT: &str = include_str!("mountexfat.ps1");

fn exfat_path() -> &'static Path {
    static DRIVE_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = tmpdir();
        let script = "mountexfat.ps1";
        let vhd = "exfat.vhd";

        // write the script file
        crate::fs::write(&dir.join(script), SCRIPT).unwrap();

        // create and mount the vhd
        let output = Command::new("pwsh.exe")
            .arg("-Command")
            .arg(format!("./{script} {vhd}",))
            .current_dir(dir.path())
            .output()
            .unwrap();

        let drive = match &output.stdout[..] {
            [drive @ b'A'..=b'Z', b'\r', b'\n'] | [drive @ b'A'..=b'Z', b'\n'] => drive,
            _ => {
                panic!("couldn't mount exfat drive\n{:?}", String::from_utf8_lossy(&output.stderr))
            }
        };

        dbg!(drive);
        PathBuf::from(format!("{drive}:\\"))
    });
    &*DRIVE_PATH
}

#[test]
fn test_exfat_please() {
    let _path = exfat_path();
}
