//@ run-pass
struct S {
    o: Option<String>
}

// Make sure we don't reuse the same alloca when matching
// on field of struct or tuple which we reassign in the match body.

fn main() {
    let mut a = (0, Some("right".to_string()));
    let b = match a.1 {
        Some(v) => {
            a.1 = Some("wrong".to_string());
            v
        }
        None => String::new()
    };
    println!("{}", b);
    assert_eq!(b, "right");


    let mut s = S{ o: Some("right".to_string()) };
    let b = match s.o {
        Some(v) => {
            s.o = Some("wrong".to_string());
            v
        }
        None => String::new(),
    };
    println!("{}", b);
    assert_eq!(b, "right");
}
