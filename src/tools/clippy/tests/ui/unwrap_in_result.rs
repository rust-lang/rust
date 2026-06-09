#![warn(clippy::unwrap_in_result)]
#![allow(clippy::ok_expect)]

struct A;

impl A {
    // should not be detected
    fn good_divisible_by_3(i_str: String) -> Result<bool, String> {
        // checks whether a string represents a number divisible by 3
        let i_result = i_str.parse::<i32>();
        match i_result {
            Err(_e) => Err("Not a number".to_string()),
            Ok(i) => {
                if i % 3 == 0 {
                    return Ok(true);
                }
                Err("Number is not divisible by 3".to_string())
            },
        }
    }

    // should be detected
    fn bad_divisible_by_3(i_str: String) -> Result<bool, String> {
        // checks whether a string represents a number divisible by 3
        let i = i_str.parse::<i32>().unwrap();
        //~^ unwrap_in_result
        if i % 3 == 0 {
            Ok(true)
        } else {
            Err("Number is not divisible by 3".to_string())
        }
    }

    fn example_option_expect(i_str: String) -> Option<bool> {
        let i = i_str.parse::<i32>().ok().expect("not a number");
        //~^ unwrap_in_result
        if i % 3 == 0 {
            return Some(true);
        }
        None
    }

    fn in_closure(a: Option<bool>) -> Option<bool> {
        // No lint inside a closure
        let c = || a.unwrap();

        // But lint outside
        let a = c().then_some(true);
        let _ = a.unwrap();
        //~^ unwrap_in_result

        None
    }

    const fn in_const_inside_fn() -> bool {
        const A: bool = {
            const fn inner(b: Option<bool>) -> Option<bool> {
                Some(b.unwrap())
                //~^ unwrap_in_result
            }

            // No lint inside `const`
            inner(Some(true)).unwrap()
        };
        A
    }

    fn in_static_inside_fn() -> bool {
        static A: bool = {
            const fn inner(b: Option<bool>) -> Option<bool> {
                Some(b.unwrap())
                //~^ unwrap_in_result
            }

            // No lint inside `static`
            inner(Some(true)).unwrap()
        };
        A
    }
}

macro_rules! mac {
    () => {
        Option::unwrap(Some(3))
    };
}

fn type_relative_unwrap() -> Option<()> {
    _ = Option::unwrap(Some(3));
    //~^ unwrap_in_result

    // Do not lint macro output
    _ = mac!();

    None
}

fn main() -> Result<(), ()> {
    A::bad_divisible_by_3("3".to_string()).unwrap();
    //~^ unwrap_in_result
    A::good_divisible_by_3("3".to_string()).unwrap();
    //~^ unwrap_in_result
    Result::unwrap(A::good_divisible_by_3("3".to_string()));
    //~^ unwrap_in_result
    Ok(())
}
