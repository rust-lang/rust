
use std::fs::{copy, read, OpenOptions};
use std::io::{SeekFrom, Result};
use std::path::PathBuf;
extern crate tempfile;
use self::tempfile::tempdir;

#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
mod test_linux {
    use super::*;
    use std::io::{Seek, Write};
    use std::process::Command;

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

    #[test]
    fn test_sparse() {
        let dir = tempdir().unwrap();
        let from = dir.path().join("sparse.bin");
        let to = dir.path().join("target.bin");

        let slen = create_sparse(&from, 0, 0).unwrap();
        assert_eq!(slen, from.metadata().unwrap().len());
        assert!(probably_sparse(&from).unwrap());

        let written = copy(&from, &to).unwrap();
        assert_eq!(slen, written);
        assert!(probably_sparse(&to).unwrap());

        let from_data = read(&from).unwrap();
        let to_data = read(&to).unwrap();
        assert_eq!(from_data, to_data);
    }

    #[test]
    fn test_sparse_leading_gap() {
        let dir = tempdir().unwrap();
        let from = dir.path().join("sparse.bin");
        let to = dir.path().join("target.bin");

        let slen = create_sparse(&from, 1024, 0).unwrap();
        assert_eq!(slen, from.metadata().unwrap().len());
        assert!(probably_sparse(&from).unwrap());

        let written = copy(&from, &to).unwrap();
        assert_eq!(slen, written);
        assert!(probably_sparse(&to).unwrap());

        assert_eq!(quickstat(&from).unwrap(), quickstat(&to).unwrap());

        let from_data = read(&from).unwrap();
        let to_data = read(&to).unwrap();
        assert_eq!(from_data, to_data);
    }

    #[test]
    fn test_sparse_trailng_gap() {
        let dir = tempdir().unwrap();
        let from = dir.path().join("sparse.bin");
        let to = dir.path().join("target.bin");

        let slen = create_sparse(&from, 1024, 1024).unwrap();
        assert_eq!(slen, from.metadata().unwrap().len());
        assert!(probably_sparse(&from).unwrap());

        let written = copy(&from, &to).unwrap();
        assert_eq!(slen, written);
        assert!(probably_sparse(&to).unwrap());
        assert_eq!(quickstat(&from).unwrap(), quickstat(&to).unwrap());

        let from_data = read(&from).unwrap();
        let to_data = read(&to).unwrap();
        assert_eq!(from_data, to_data);
    }

    #[test]
    fn test_empty_sparse() {
        let dir = tempdir().unwrap();
        let from = dir.path().join("sparse.bin");
        let to = dir.path().join("target.bin");

        let out = Command::new("/usr/bin/truncate")
            .args(&["-s", "1M", from.to_str().unwrap()])
            .output().unwrap();
        assert!(out.status.success());
        assert_eq!(from.metadata().unwrap().len(), 1024*1024);

        let _written = copy(&from, &to).unwrap();
        assert_eq!(to.metadata().unwrap().len(), 1024*1024);

        assert!(probably_sparse(&to).unwrap());
        assert_eq!(quickstat(&from).unwrap(), quickstat(&to).unwrap());

        let from_data = read(&from).unwrap();
        let to_data = read(&to).unwrap();
        assert_eq!(from_data, to_data);
    }

}
