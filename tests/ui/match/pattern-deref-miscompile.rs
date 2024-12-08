//@ run-pass

fn main() {
    match b"." as &[u8] {
        b"." if true => {},
        b"." => panic!(),
        b".." => panic!(),
        b"" => panic!(),
        _ => panic!(),
    }
    match b"." as &[u8] {
        b"." if false => panic!(),
        b"." => {},
        b".." => panic!(),
        b"" => panic!(),
        _ => panic!(),
    }
    match b".." as &[u8] {
        b"." if true => panic!(), // the miscompile caused this arm to be reached
        b"." => panic!(),
        b".." => {},
        b"" => panic!(),
        _ => panic!(),
    }
    match b".." as &[u8] {
        b"." if false => panic!(),
        b"." => panic!(),
        b".." => {},
        b"" => panic!(),
        _ => panic!(),
    }
    match b"" as &[u8] {
        b"." if true => panic!(),
        b"." => panic!(),
        b".." => panic!(),
        b"" => {},
        _ => panic!(),
    }
    match b"" as &[u8] {
        b"." if false => panic!(),
        b"." => panic!(),
        b".." => panic!(),
        b"" => {},
        _ => panic!(),
    }
}
