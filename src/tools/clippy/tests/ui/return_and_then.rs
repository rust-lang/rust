#![warn(clippy::return_and_then)]

fn main() {
    fn test_opt_block(opt: Option<i32>) -> Option<i32> {
        opt.and_then(|n| {
            let mut ret = n + 1;
            ret += n;
            if n > 1 { Some(ret) } else { None }
        })
    }

    fn test_opt_func(opt: Option<i32>) -> Option<i32> {
        opt.and_then(|n| test_opt_block(Some(n)))
    }

    fn test_call_chain() -> Option<i32> {
        gen_option(1).and_then(|n| test_opt_block(Some(n)))
    }

    fn test_res_block(opt: Result<i32, i32>) -> Result<i32, i32> {
        opt.and_then(|n| if n > 1 { Ok(n + 1) } else { Err(n) })
    }

    fn test_res_func(opt: Result<i32, i32>) -> Result<i32, i32> {
        opt.and_then(|n| test_res_block(Ok(n)))
    }

    fn test_ref_only() -> Option<i32> {
        // ref: empty string
        Some("").and_then(|x| if x.len() > 2 { Some(3) } else { None })
    }

    fn test_tmp_only() -> Option<i32> {
        // unused temporary: vec![1, 2, 4]
        Some(match (vec![1, 2, 3], vec![1, 2, 4]) {
            (a, _) if a.len() > 1 => a,
            (_, b) => b,
        })
        .and_then(|x| if x.len() > 2 { Some(3) } else { None })
    }

    // should not lint
    fn test_tmp_ref() -> Option<String> {
        String::from("<BOOM>")
            .strip_prefix("<")
            .and_then(|s| s.strip_suffix(">").map(String::from))
    }

    // should not lint
    fn test_unconsumed_tmp() -> Option<i32> {
        [1, 2, 3]
            .iter()
            .map(|x| x + 1)
            .collect::<Vec<_>>() // temporary Vec created here
            .as_slice() // creates temporary slice
            .first() // creates temporary reference
            .and_then(|x| test_opt_block(Some(*x)))
    }
}

fn gen_option(n: i32) -> Option<i32> {
    Some(n)
}
