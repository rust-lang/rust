
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Read, Write, Result};
use std::path::PathBuf;


#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
mod test_linux {
    use super::*;
    use std::process::{Command, Output};

    fn create_sparse(file: &PathBuf, head: u64, tail: u64) -> Result<u64> {
        let data = "c00lc0d3";
        let len = 4096u64 * 4096 + data.len() as u64 + tail;

        let out = Command::new("truncate")
            .args(&["-s", len.to_string().as_str(),
                    file.to_str().unwrap()])
            .output()?;
        assert!(out.status.success());

        let mut fd = OpenOptions::new()
            .write(true)
            .append(false)
            .open(&file)?;

        fd.seek(SeekFrom::Start(head))?;
        write!(fd, "{}", data);

        fd.seek(SeekFrom::Start(1024*4096))?;
        write!(fd, "{}", data);

        fd.seek(SeekFrom::Start(4096*4096))?;
        write!(fd, "{}", data);

        Ok(len as u64)
    }

    fn quickstat(file: &PathBuf) -> Result<(i32, i32, i32)> {
        let out = Command::new("stat")
            .args(&["--format", "%s %b %B",
                    file.to_str().unwrap()])
            .output()?;
        assert!(out.status.success());

        let stdout = String::from_utf8(out.stdout).unwrap();
        let stats = stdout
            .split_whitespace()
            .map(|s| s.parse::<i32>().unwrap())
            .collect::<Vec<i32>>();
        let (size, blocks, blksize) = (stats[0], stats[1], stats[2]);

        Ok((size, blocks, blksize))
    }

    fn probably_sparse(file: &PathBuf) -> Result<bool> {
        let (size, blocks, blksize) = quickstat(file)?;

        Ok(blocks < size / blksize)
    }


    #[test]
    fn test_tests() {
        assert!(true);
    }
}
