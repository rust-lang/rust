// check-pass

fn let_underscore(string: &Option<&str>, mut num: Option<i32>) {
    let _ = if let Some(s) = *string { s.len() } else { 0 };
    let _ = if let Some(s) = &num { s } else { &0 };
    let _ = if let Some(s) = &mut num {
        *s += 1;
        s
    } else {
        &mut 0
    };
    let _ = if let Some(ref s) = num { s } else { &0 };
    let _ = if let Some(mut s) = num {
        s += 1;
        s
    } else {
        0
    };
    let _ = if let Some(ref mut s) = num {
        *s += 1;
        s
    } else {
        &mut 0
    };
}

fn main() {}
