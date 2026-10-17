//@ check-fail
//@ forbid-output: backend2.push(
use std::path::PathBuf;

type Test = Option<(PathBuf, Vec<u8>)>;

fn test1(backend: &mut Vec<Test>) -> Result<(), String> {
    let mut backend2 = Vec::new();
    for b in backend {
        backend2.push(match b {
            None => None,
            Some((file, bytes)) => Some((file.as_path(), bytes)),
        });
    }

    loop {
        test2(&mut backend2)?; //~ ERROR E0499
    }
}

fn test2<'a, P, C>(disks: &'a mut [Option<(P, &'a mut Vec<u8>)>]) -> Result<C, String> {
    todo!()
}

fn test3<'a, P, C>(disks: &'a mut [Option<(P, &'a mut Vec<u8>)>]) -> Result<C, String> {
    loop {
        test2(disks)?; //~ ERROR E0499
    }
}

fn main() {}
