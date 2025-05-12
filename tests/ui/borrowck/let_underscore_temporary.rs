//@ check-fail

fn let_underscore(string: &Option<&str>, mut num: Option<i32>) {
    let _ = if let Some(s) = *string { s.len() } else { 0 };
    let _ = if let Some(s) = &num { s } else { &0 };
    let _ = if let Some(s) = &mut num {
        *s += 1;
        s
    } else {
        let a = 0;
        &a
        //~^ ERROR does not live long enough
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
        let a = 0;
        &a
        //~^ ERROR does not live long enough
    };
}

fn let_ascribe(string: &Option<&str>, mut num: Option<i32>) {
    let _: _ = if let Some(s) = *string { s.len() } else { 0 };
    let _: _ = if let Some(s) = &num { s } else { &0 };
    let _: _ = if let Some(s) = &mut num {
        *s += 1;
        s
    } else {
        let a = 0;
        &a
        //~^ ERROR does not live long enough
    };
    let _: _ = if let Some(ref s) = num { s } else { &0 };
    let _: _ = if let Some(mut s) = num {
        s += 1;
        s
    } else {
        0
    };
    let _: _ = if let Some(ref mut s) = num {
        *s += 1;
        s
    } else {
        let a = 0;
        &a
        //~^ ERROR does not live long enough
    };
}

fn matched(string: &Option<&str>, mut num: Option<i32>) {
    match if let Some(s) = *string { s.len() } else { 0 } {
        _ => {}
    };
    match if let Some(s) = &num { s } else { &0 } {
        _ => {}
    };
    match if let Some(s) = &mut num {
        *s += 1;
        s
    } else {
        let a = 0;
        &a
        //~^ ERROR does not live long enough
    } {
        _ => {}
    };
    match if let Some(ref s) = num { s } else { &0 } {
        _ => {}
    };
    match if let Some(mut s) = num {
        s += 1;
        s
    } else {
        0
    } {
        _ => {}
    };
    match if let Some(ref mut s) = num {
        *s += 1;
        s
    } else {
        let a = 0;
        &a
        //~^ ERROR does not live long enough
    } {
        _ => {}
    };
}

fn main() {}
