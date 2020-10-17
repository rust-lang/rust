#![warn(clippy::explicit_counter_loop)]

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
