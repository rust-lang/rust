// run-pass

const FOO: &[u8] = b"foo";
const BAR: &[u8] = &[1, 2, 3];

const BOO: &i32 = &42;

fn main() {
    match &[1u8, 2, 3] as &[u8] {
        FOO => panic!("a"),
        BAR => println!("b"),
        _ => panic!("c"),
    }

    match b"foo" as &[u8] {
        FOO => println!("a"),
        BAR => panic!("b"),
        _ => panic!("c"),
    }

    #[allow(unreachable_patterns)]
    match &43 {
        &42 => panic!(),
        BOO => panic!(),
        _ => println!("d"),
    }
}
