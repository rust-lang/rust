// run-pass
// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes

// Tests ensuring that `dbg!(expr)` has the expected run-time behavior.
// as well as some compile time properties we expect.

#[derive(Copy, Clone, Debug)]
struct Unit;

#[derive(Copy, Clone, Debug, PartialEq)]
struct Point<T> {
    x: T,
    y: T,
}

#[derive(Debug, PartialEq)]
struct NoCopy(usize);

fn test() {
    let a: Unit = dbg!(Unit);
    let _: Unit = dbg!(a);
    // We can move `a` because it's Copy.
    drop(a);

    // `Point<T>` will be faithfully formatted according to `{:#?}`.
    let a = Point { x: 42, y: 24 };
    let b: Point<u8> = dbg!(Point { x: 42, y: 24 }); // test stringify!(..)
    let c: Point<u8> = dbg!(b);
    // Identity conversion:
    assert_eq!(a, b);
    assert_eq!(a, c);
    // We can move `b` because it's Copy.
    drop(b);

    // Without parameters works as expected.
    let _: () = dbg!();

    // Test that we can borrow and that successive applications is still identity.
    let a = NoCopy(1337);
    let b: &NoCopy = dbg!(dbg!(&a));
    assert_eq!(&a, b);

    // Test involving lifetimes of temporaries:
    fn f<'a>(x: &'a u8) -> &'a u8 { x }
    let a: &u8 = dbg!(f(&42));
    assert_eq!(a, &42);

    // Test side effects:
    let mut foo = 41;
    assert_eq!(7331, dbg!({
        foo += 1;
        eprintln!("before");
        7331
    }));
    assert_eq!(foo, 42);

    // Test trailing comma:
    assert_eq!(("Yeah",), dbg!(("Yeah",)));

    // Test multiple arguments:
    assert_eq!((1u8, 2u32), dbg!(1,
                                 2));

    // Test multiple arguments + trailing comma:
    assert_eq!((1u8, 2u32, "Yeah"), dbg!(1u8, 2u32,
                                         "Yeah",));
}

fn validate_stderr(stderr: Vec<String>) {
    assert_eq!(stderr, &[
        ":22] Unit = Unit",

        ":23] a = Unit",

        ":29] Point{x: 42, y: 24,} = Point {",
        "    x: 42,",
        "    y: 24,",
        "}",

        ":30] b = Point {",
        "    x: 42,",
        "    y: 24,",
        "}",

        ":38]",

        ":42] &a = NoCopy(",
        "    1337,",
        ")",

        ":42] dbg!(& a) = NoCopy(",
        "    1337,",
        ")",
        ":47] f(&42) = 42",

        "before",
        ":52] { foo += 1; eprintln!(\"before\"); 7331 } = 7331",

        ":60] (\"Yeah\",) = (",
        "    \"Yeah\",",
        ")",

        ":63] 1 = 1",
        ":63] 2 = 2",

        ":67] 1u8 = 1",
        ":67] 2u32 = 2",
        ":67] \"Yeah\" = \"Yeah\"",
    ]);
}

fn main() {
    // The following is a hack to deal with compiletest's inability
    // to check the output (to stdout) of run-pass tests.
    use std::env;
    use std::process::Command;

    let mut args = env::args();
    let prog = args.next().unwrap();
    let child = args.next();
    if let Some("child") = child.as_ref().map(|s| &**s) {
        // Only run the test if we've been spawned as 'child'
        test()
    } else {
        // This essentially spawns as 'child' to run the tests
        // and then it collects output of stderr and checks the output
        // against what we expect.
        let out = Command::new(&prog).arg("child").output().unwrap();
        assert!(out.status.success());
        assert!(out.stdout.is_empty());

        let stderr = String::from_utf8(out.stderr).unwrap();
        let stderr = stderr.lines().map(|mut s| {
            if s.starts_with("[") {
                // Strip `[` and file path:
                s = s.trim_start_matches("[");
                assert!(s.starts_with(file!()));
                s = s.trim_start_matches(file!());
            }
            s.to_owned()
        }).collect();

        validate_stderr(stderr);
    }
}
