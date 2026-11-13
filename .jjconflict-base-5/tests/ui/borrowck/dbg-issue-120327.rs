fn s() -> String {
    let a = String::new();
    dbg!(a);
    return a; //~ ERROR use of moved value:
}

fn m() -> String {
    let a = String::new();
    dbg!(1, 2, a, 1, 2);
    return a; //~ ERROR use of moved value:
}

fn t(a: String) -> String {
    let b: String = "".to_string();
    dbg!(a, b);
    return b; //~ ERROR use of moved value:
}

fn x(a: String) -> String {
    let b: String = "".to_string();
    dbg!(a, b);
    return a; //~ ERROR use of moved value:
}

macro_rules! my_dbg {
    () => {
        eprintln!("[{}:{}:{}]", file!(), line!(), column!())
    };
    ($val:expr $(,)?) => {
        match $val {
            tmp => {
                eprintln!("[{}:{}:{}] {} = {:#?}",
                    file!(), line!(), column!(), stringify!($val), &tmp);
                tmp
            }
        }
    };
    ($($val:expr),+ $(,)?) => {
        ($(my_dbg!($val)),+,)
    };
}

fn test_my_dbg() -> String {
    let b: String = "".to_string();
    my_dbg!(b, 1);
    return b; //~ ERROR use of moved value:
}

fn test_not_macro() -> String {
    let a = String::new();
    let _b = match a {
        tmp => {
            eprintln!("dbg: {}", tmp);
            tmp
        }
    };
    return a; //~ ERROR use of moved value:
}

fn get_expr(_s: String) {}

fn test() {
    let a: String = "".to_string();
    let _res = get_expr(dbg!(a));
    let _l = a.len(); //~ ERROR borrow of moved value
}

fn main() {}
