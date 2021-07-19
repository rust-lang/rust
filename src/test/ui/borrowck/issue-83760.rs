struct Struct;

fn test1() {
    let mut val = Some(Struct);
    while let Some(foo) = val { //~ ERROR use of moved value
        if true {
            val = None;
        } else {

        }
    }
}

fn test2() {
    let mut foo = Some(Struct);
    let _x = foo.unwrap();
    if true {
        foo = Some(Struct);
    } else {
    }
    let _y = foo; //~ ERROR use of moved value: `foo`
}

fn test3() {
    let mut foo = Some(Struct);
    let _x = foo.unwrap();
    if true {
        foo = Some(Struct);
    } else if true {
        foo = Some(Struct);
    } else if true {
        foo = Some(Struct);
    } else if true {
        foo = Some(Struct);
    } else {
    }
    let _y = foo; //~ ERROR use of moved value: `foo`
}

fn main() {}
