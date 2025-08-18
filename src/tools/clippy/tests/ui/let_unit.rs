#![warn(clippy::let_unit_value)]
#![allow(unused, clippy::no_effect, clippy::needless_late_init, path_statements)]

macro_rules! let_and_return {
    ($n:expr) => {{
        let ret = $n;
    }};
}

fn main() {
    let _x = println!("x");
    //~^ let_unit_value
    let _y = 1; // this is fine
    let _z = ((), 1); // this as well
    if true {
        // do not lint this, since () is explicit
        let _a = ();
        let () = dummy();
        let () = ();
        () = dummy();
        () = ();
        let _a: () = ();
        let _a: () = dummy();
    }

    consume_units_with_for_loop(); // should be fine as well

    multiline_sugg();

    let_and_return!(()) // should be fine
}

fn dummy() {}

// Related to issue #1964
fn consume_units_with_for_loop() {
    // `for_let_unit` lint should not be triggered by consuming them using for loop.
    let v = vec![(), (), ()];
    let mut count = 0;
    for _ in v {
        count += 1;
    }
    assert_eq!(count, 3);

    // Same for consuming from some other Iterator<Item = ()>.
    let (tx, rx) = ::std::sync::mpsc::channel();
    tx.send(()).unwrap();
    drop(tx);

    count = 0;
    for _ in rx.iter() {
        count += 1;
    }
    assert_eq!(count, 1);
}

fn multiline_sugg() {
    let v: Vec<u8> = vec![2];

    let _ = v
        //~^ let_unit_value
        .into_iter()
        .map(|i| i * 2)
        .filter(|i| i.is_multiple_of(2))
        .map(|_| ())
        .next()
        .unwrap();
}

#[derive(Copy, Clone)]
pub struct ContainsUnit(()); // should be fine

fn _returns_generic() {
    fn f<T>() -> T {
        unimplemented!()
    }
    fn f2<T, U>(_: T) -> U {
        unimplemented!()
    }
    fn f3<T>(x: T) -> T {
        x
    }
    fn f5<T: Default>(x: bool) -> Option<T> {
        x.then(|| T::default())
    }

    let _: () = f();
    let x: () = f();

    let _: () = f2(0i32);
    let x: () = f2(0i32);

    let _: () = f3(());
    let x: () = f3(());

    fn f4<T>(mut x: Vec<T>) -> T {
        x.pop().unwrap()
    }
    let _: () = f4(vec![()]);
    let x: () = f4(vec![()]);

    let _: () = {
        let x = 5;
        f2(x)
    };

    let _: () = if true { f() } else { f2(0) };
    let x: () = if true { f() } else { f2(0) };

    let x = match Some(0) {
        //~^ let_unit_value
        None => f2(1),
        Some(0) => f(),
        Some(1) => f2(3),
        Some(_) => (),
    };

    let _: () = f5(true).unwrap();

    #[allow(clippy::let_unit_value)]
    {
        let x = f();
        let y;
        let z;
        match 0 {
            0 => {
                y = f();
                z = f();
            },
            1 => {
                println!("test");
                y = f();
                z = f3(());
            },
            _ => panic!(),
        }

        let x1;
        let x2;
        if true {
            x1 = f();
            x2 = x1;
        } else {
            x2 = f();
            x1 = x2;
        }

        let opt;
        match f5(true) {
            Some(x) => opt = x,
            None => panic!(),
        };

        #[warn(clippy::let_unit_value)]
        {
            let _: () = x;
            let _: () = y;
            let _: () = z;
            let _: () = x1;
            let _: () = x2;
            let _: () = opt;
        }
    }

    let () = f();
}

fn attributes() {
    fn f() {}

    #[allow(clippy::let_unit_value)]
    let _ = f();
    #[expect(clippy::let_unit_value)]
    let _ = f();
}

async fn issue10433() {
    let _pending: () = std::future::pending().await;
}

pub async fn issue11502(a: ()) {}

pub fn issue12594() {
    fn returns_unit() {}

    fn returns_result<T>(res: T) -> Result<T, ()> {
        Ok(res)
    }

    fn actual_test() {
        // create first a unit value'd value
        let res = returns_unit();
        //~^ let_unit_value
        returns_result(res).unwrap();
        returns_result(res).unwrap();
        // make sure we replace only the first variable
        let res = 1;
        returns_result(res).unwrap();
    }
}

fn issue15061() {
    fn return_unit() {}
    fn do_something(x: ()) {}

    let res = return_unit();
    //~^ let_unit_value
    do_something(res);
    println!("{res:?}");
}
