fn main() {
    let x = Some(());
    if x.is_some() {
        x.unwrap();
    }
    if x.is_none() {
        // nothing to do here
    } else {
        x.unwrap();
    }
    let mut x: Result<(), ()> = Ok(());
    if x.is_ok() {
        x.unwrap();
    } else {
        x.unwrap_err();
    }
    if x.is_err() {
        x.unwrap_err();
    } else {
        x.unwrap();
    }
    if x.is_ok() {
        x = Err(());
        x.unwrap();
    } else {
        x = Ok(());
        x.unwrap_err();
    }
}

fn test_complex_conditions() {
    let x: Result<(), ()> = Ok(());
    let y: Result<(), ()> = Ok(());
    if x.is_ok() && y.is_err() {
        x.unwrap();
        y.unwrap_err();
    } else {
        // not clear whether unwrappable:
        x.unwrap_err();
        y.unwrap();
    }

    if x.is_ok() || y.is_ok() {
        // not clear whether unwrappable:
        x.unwrap();
        y.unwrap();
    } else {
        x.unwrap_err();
        y.unwrap_err();
    }
    let z: Result<(), ()> = Ok(());
    if x.is_ok() && !(y.is_ok() || z.is_err()) {
        x.unwrap();
        y.unwrap_err();
        z.unwrap();
    }
    if x.is_ok() || !(y.is_ok() && z.is_err()) {
        // not clear what's unwrappable
    } else {
        x.unwrap_err();
        y.unwrap();
        z.unwrap_err();
    }
}

fn test_nested() {
    fn nested() {
        let x = Some(());
        if x.is_some() {
            x.unwrap();
        }
    }
}
