#![deny(clippy::bind_instead_of_map)]
#![allow(clippy::blocks_in_conditions)]

pub fn main() {
    let _ = Some("42").and_then(|s| if s.len() < 42 { Some(0) } else { Some(s.len()) });
    let _ = Some("42").and_then(|s| if s.len() < 42 { None } else { Some(s.len()) });

    let _ = Ok::<_, ()>("42").and_then(|s| if s.len() < 42 { Ok(0) } else { Ok(s.len()) });
    let _ = Ok::<_, ()>("42").and_then(|s| if s.len() < 42 { Err(()) } else { Ok(s.len()) });

    let _ = Err::<(), _>("42").or_else(|s| if s.len() < 42 { Err(s.len() + 20) } else { Err(s.len()) });
    let _ = Err::<(), _>("42").or_else(|s| if s.len() < 42 { Ok(()) } else { Err(s.len()) });

    hard_example();
    macro_example();
}

fn hard_example() {
    Some("42").and_then(|s| {
        if {
            if s == "43" {
                return Some(43);
            }
            s == "42"
        } {
            return Some(45);
        }
        match s.len() {
            10 => Some(2),
            20 => {
                if foo() {
                    return {
                        if foo() {
                            return Some(20);
                        }
                        println!("foo");
                        Some(3)
                    };
                }
                Some(20)
            },
            40 => Some(30),
            _ => Some(1),
        }
    });
}

fn foo() -> bool {
    true
}

macro_rules! m {
    () => {
        Some(10)
    };
}

fn macro_example() {
    let _ = Some("").and_then(|s| if s.len() == 20 { m!() } else { Some(20) });
    let _ = Some("").and_then(|s| if s.len() == 20 { Some(m!()) } else { Some(Some(20)) });
}
