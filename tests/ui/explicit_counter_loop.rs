#![warn(clippy::explicit_counter_loop)]
#![allow(clippy::uninlined_format_args, clippy::useless_vec)]

fn main() {
    let mut vec = vec![1, 2, 3, 4];
    let mut _index = 0;
    for _v in &vec {
        _index += 1
    }

    let mut _index = 1;
    _index = 0;
    for _v in &vec {
        _index += 1
    }

    let mut _index = 0;
    for _v in &mut vec {
        _index += 1;
    }

    let mut _index = 0;
    for _v in vec {
        _index += 1;
    }

    let vec = [1, 2, 3, 4];
    // Potential false positives
    let mut _index = 0;
    _index = 1;
    for _v in &vec {
        _index += 1
    }

    let mut _index = 0;
    _index += 1;
    for _v in &vec {
        _index += 1
    }

    let mut _index = 0;
    for _v in &vec {
        _index = 1;
        _index += 1
    }

    let mut _index = 0;
    for _v in &vec {
        let mut _index = 0;
        _index += 1
    }

    let mut _index = 0;
    for _v in &vec {
        _index += 1;
        _index = 0;
    }

    let mut _index = 0;
    if true {
        _index = 1
    };
    for _v in &vec {
        _index += 1
    }

    let mut _index = 1;
    if false {
        _index = 0
    };
    for _v in &vec {
        _index += 1
    }
}

mod issue_1219 {
    pub fn test() {
        // should not trigger the lint because variable is used after the loop #473
        let vec = vec![1, 2, 3];
        let mut index = 0;
        for _v in &vec {
            index += 1
        }
        println!("index: {}", index);

        // should not trigger the lint because the count is conditional #1219
        let text = "banana";
        let mut count = 0;
        for ch in text.chars() {
            println!("{}", count);
            if ch == 'a' {
                continue;
            }
            count += 1;
        }

        // should not trigger the lint because the count is conditional
        let text = "banana";
        let mut count = 0;
        for ch in text.chars() {
            println!("{}", count);
            if ch == 'a' {
                count += 1;
            }
        }

        // should trigger the lint because the count is not conditional
        let text = "banana";
        let mut count = 0;
        for ch in text.chars() {
            println!("{}", count);
            count += 1;
            if ch == 'a' {
                continue;
            }
        }

        // should trigger the lint because the count is not conditional
        let text = "banana";
        let mut count = 0;
        for ch in text.chars() {
            println!("{}", count);
            count += 1;
            for i in 0..2 {
                let _ = 123;
            }
        }

        // should not trigger the lint because the count is incremented multiple times
        let text = "banana";
        let mut count = 0;
        for ch in text.chars() {
            println!("{}", count);
            count += 1;
            for i in 0..2 {
                count += 1;
            }
        }
    }
}

mod issue_3308 {
    pub fn test() {
        // should not trigger the lint because the count is incremented multiple times
        let mut skips = 0;
        let erasures = vec![];
        for i in 0..10 {
            println!("{}", skips);
            while erasures.contains(&(i + skips)) {
                skips += 1;
            }
        }

        // should not trigger the lint because the count is incremented multiple times
        let mut skips = 0;
        for i in 0..10 {
            println!("{}", skips);
            let mut j = 0;
            while j < 5 {
                skips += 1;
                j += 1;
            }
        }

        // should not trigger the lint because the count is incremented multiple times
        let mut skips = 0;
        for i in 0..10 {
            println!("{}", skips);
            for j in 0..5 {
                skips += 1;
            }
        }
    }
}

mod issue_1670 {
    pub fn test() {
        let mut count = 0;
        for _i in 3..10 {
            count += 1;
        }
    }
}

mod issue_4732 {
    pub fn test() {
        let slice = &[1, 2, 3];
        let mut index = 0;

        // should not trigger the lint because the count is used after the loop
        for _v in slice {
            index += 1
        }
        let _closure = || println!("index: {}", index);
    }
}

mod issue_4677 {
    pub fn test() {
        let slice = &[1, 2, 3];

        // should not trigger the lint because the count is used after incremented
        let mut count = 0;
        for _i in slice {
            count += 1;
            println!("{}", count);
        }
    }
}

mod issue_7920 {
    pub fn test() {
        let slice = &[1, 2, 3];

        let index_usize: usize = 0;
        let mut idx_usize: usize = 0;

        // should suggest `enumerate`
        for _item in slice {
            if idx_usize == index_usize {
                break;
            }

            idx_usize += 1;
        }

        let index_u32: u32 = 0;
        let mut idx_u32: u32 = 0;

        // should suggest `zip`
        for _item in slice {
            if idx_u32 == index_u32 {
                break;
            }

            idx_u32 += 1;
        }
    }
}

mod issue_10058 {
    pub fn test() {
        // should not lint since we are increasing counter potentially more than once in the loop
        let values = [0, 1, 0, 1, 1, 1, 0, 1, 0, 1];
        let mut counter = 0;
        for value in values {
            counter += 1;

            if value == 0 {
                continue;
            }

            counter += 1;
        }
    }

    pub fn test2() {
        // should not lint since we are increasing counter potentially more than once in the loop
        let values = [0, 1, 0, 1, 1, 1, 0, 1, 0, 1];
        let mut counter = 0;
        for value in values {
            counter += 1;

            if value != 0 {
                counter += 1;
            }
        }
    }
}
