//@ edition:2021
//@ run-pass

// Test precise capture when using boxes

struct MetaData { x: String, name: String }
struct Data { m: MetaData }
struct BoxedData(Box<Data>);
struct EvenMoreBoxedData(Box<BoxedData>);

// Mutate disjoint paths, one inside one outside the closure
fn box_1() {
    let m = MetaData { x: format!("x"), name: format!("name") };
    let d = Data { m };
    let b = BoxedData(Box::new(d));
    let mut e = EvenMoreBoxedData(Box::new(b));

    let mut c = || {
        e.0.0.m.x = format!("not-x");
    };

    e.0.0.m.name = format!("not-name");
    c();
}

// Mutate a path inside the closure and read a disjoint path outside the closure
fn box_2() {
    let m = MetaData { x: format!("x"), name: format!("name") };
    let d = Data { m };
    let b = BoxedData(Box::new(d));
    let mut e = EvenMoreBoxedData(Box::new(b));

    let mut c = || {
        e.0.0.m.x = format!("not-x");
    };

    println!("{}", e.0.0.m.name);
    c();
}

// Read a path inside the closure and mutate a disjoint path outside the closure
fn box_3() {
    let m = MetaData { x: format!("x"), name: format!("name") };
    let d = Data { m };
    let b = BoxedData(Box::new(d));
    let mut e = EvenMoreBoxedData(Box::new(b));

    let c = || {
        println!("{}", e.0.0.m.name);
    };

    e.0.0.m.x = format!("not-x");
    c();
}

// Read disjoint paths, one inside the closure and one outside the closure.
fn box_4() {
    let m = MetaData { x: format!("x"), name: format!("name") };
    let d = Data { m };
    let b = BoxedData(Box::new(d));
    let e = EvenMoreBoxedData(Box::new(b));

    let c = || {
        println!("{}", e.0.0.m.name);
    };

    println!("{}", e.0.0.m.x);
    c();
}

// Read the same path, once inside the closure and once outside the closure.
fn box_5() {
    let m = MetaData { x: format!("x"), name: format!("name") };
    let d = Data { m };
    let b = BoxedData(Box::new(d));
    let e = EvenMoreBoxedData(Box::new(b));

    let c = || {
        println!("{}", e.0.0.m.name);
    };

    println!("{}", e.0.0.m.name);
    c();
}

fn main() {
    box_1();
    box_2();
    box_3();
    box_4();
    box_5();
}
