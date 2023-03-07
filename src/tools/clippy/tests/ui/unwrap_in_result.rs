#![warn(clippy::unwrap_in_result)]

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
        if i % 3 == 0 {
            Ok(true)
        } else {
            Err("Number is not divisible by 3".to_string())
        }
    }

    fn example_option_expect(i_str: String) -> Option<bool> {
        let i = i_str.parse::<i32>().expect("not a number");
        if i % 3 == 0 {
            return Some(true);
        }
        None
    }
}

fn main() {
    A::bad_divisible_by_3("3".to_string());
    A::good_divisible_by_3("3".to_string());
}
