fn add(a: i32, b: i32) -> i32 {
    a + b
}
fn main() {
    // We shouldn't coerce capturing closure to a function
    let cap = 0;
    let _ = match "+" {
        "+" => add,
        "-" => |a, b| (a - b + cap) as i32,
        _ => unimplemented!(),
    };
    //~^^^ ERROR `match` arms have incompatible types


    // We shouldn't coerce capturing closure to a non-capturing closure
    let _ = match "+" {
        "+" => |a, b| (a + b) as i32,
        "-" => |a, b| (a - b + cap) as i32,
        _ => unimplemented!(),
    };
    //~^^^ ERROR `match` arms have incompatible types


    // We shouldn't coerce non-capturing closure to a capturing closure
    let _ = match "+" {
        "+" => |a, b| (a + b + cap) as i32,
        "-" => |a, b| (a - b) as i32,
        _ => unimplemented!(),
    };
    //~^^^ ERROR `match` arms have incompatible types

    // We shouldn't coerce capturing closure to a capturing closure
    let _ = match "+" {
        "+" => |a, b| (a + b + cap) as i32,
        "-" => |a, b| (a - b + cap) as i32,
        _ => unimplemented!(),
    };
    //~^^^ ERROR `match` arms have incompatible types
}
