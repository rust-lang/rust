enum Cause { Cause1, Cause2 }
struct MyErr { x: Cause }

fn main() {
    _ = f();
}

fn f() -> Result<i32, MyErr> {
    let res = could_fail();
    let x = if let Ok(x) = res {
        x
    } else if let Err(e) = res { //~ ERROR `if` and `else`
        return Err(e);
    };
    Ok(x)
}

fn could_fail() -> Result<i32, MyErr> {
    Ok(0)
}
