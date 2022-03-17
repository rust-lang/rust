use super::*;

use std::result::Result::Ok;

#[test]
fn test_basic_setabf() {
    let s = b"\\E[48;5;%p1%dm";
    assert_eq!(
        expand(s, &[Number(1)], &mut Variables::new()).unwrap(),
        "\\E[48;5;1m".bytes().collect::<Vec<_>>()
    );
}

#[test]
fn test_multiple_int_constants() {
    assert_eq!(
        expand(b"%{1}%{2}%d%d", &[], &mut Variables::new()).unwrap(),
        "21".bytes().collect::<Vec<_>>()
    );
}

#[test]
fn test_op_i() {
    let mut vars = Variables::new();
    assert_eq!(
        expand(b"%p1%d%p2%d%p3%d%i%p1%d%p2%d%p3%d", &[Number(1), Number(2), Number(3)], &mut vars),
        Ok("123233".bytes().collect::<Vec<_>>())
    );
    assert_eq!(
        expand(b"%p1%d%p2%d%i%p1%d%p2%d", &[], &mut vars),
        Ok("0011".bytes().collect::<Vec<_>>())
    );
}

#[test]
fn test_param_stack_failure_conditions() {
    let mut varstruct = Variables::new();
    let vars = &mut varstruct;
    fn get_res(
        fmt: &str,
        cap: &str,
        params: &[Param],
        vars: &mut Variables,
    ) -> Result<Vec<u8>, String> {
        let mut u8v: Vec<_> = fmt.bytes().collect();
        u8v.extend(cap.as_bytes().iter().map(|&b| b));
        expand(&u8v, params, vars)
    }

    let caps = ["%d", "%c", "%s", "%Pa", "%l", "%!", "%~"];
    for &cap in caps.iter() {
        let res = get_res("", cap, &[], vars);
        assert!(res.is_err(), "Op {} succeeded incorrectly with 0 stack entries", cap);
        if cap == "%s" || cap == "%l" {
            continue;
        }
        let p = Number(97);
        let res = get_res("%p1", cap, &[p], vars);
        assert!(res.is_ok(), "Op {} failed with 1 stack entry: {}", cap, res.unwrap_err());
    }
    let caps = ["%+", "%-", "%*", "%/", "%m", "%&", "%|", "%A", "%O"];
    for &cap in caps.iter() {
        let res = expand(cap.as_bytes(), &[], vars);
        assert!(res.is_err(), "Binop {} succeeded incorrectly with 0 stack entries", cap);
        let res = get_res("%{1}", cap, &[], vars);
        assert!(res.is_err(), "Binop {} succeeded incorrectly with 1 stack entry", cap);
        let res = get_res("%{1}%{2}", cap, &[], vars);
        assert!(res.is_ok(), "Binop {} failed with 2 stack entries: {}", cap, res.unwrap_err());
    }
}

#[test]
fn test_push_bad_param() {
    assert!(expand(b"%pa", &[], &mut Variables::new()).is_err());
}

#[test]
fn test_comparison_ops() {
    let v = [('<', [1u8, 0u8, 0u8]), ('=', [0u8, 1u8, 0u8]), ('>', [0u8, 0u8, 1u8])];
    for &(op, bs) in v.iter() {
        let s = format!("%{{1}}%{{2}}%{}%d", op);
        let res = expand(s.as_bytes(), &[], &mut Variables::new());
        assert!(res.is_ok(), "{}", res.unwrap_err());
        assert_eq!(res.unwrap(), vec![b'0' + bs[0]]);
        let s = format!("%{{1}}%{{1}}%{}%d", op);
        let res = expand(s.as_bytes(), &[], &mut Variables::new());
        assert!(res.is_ok(), "{}", res.unwrap_err());
        assert_eq!(res.unwrap(), vec![b'0' + bs[1]]);
        let s = format!("%{{2}}%{{1}}%{}%d", op);
        let res = expand(s.as_bytes(), &[], &mut Variables::new());
        assert!(res.is_ok(), "{}", res.unwrap_err());
        assert_eq!(res.unwrap(), vec![b'0' + bs[2]]);
    }
}

#[test]
fn test_conditionals() {
    let mut vars = Variables::new();
    let s = b"\\E[%?%p1%{8}%<%t3%p1%d%e%p1%{16}%<%t9%p1%{8}%-%d%e38;5;%p1%d%;m";
    let res = expand(s, &[Number(1)], &mut vars);
    assert!(res.is_ok(), "{}", res.unwrap_err());
    assert_eq!(res.unwrap(), "\\E[31m".bytes().collect::<Vec<_>>());
    let res = expand(s, &[Number(8)], &mut vars);
    assert!(res.is_ok(), "{}", res.unwrap_err());
    assert_eq!(res.unwrap(), "\\E[90m".bytes().collect::<Vec<_>>());
    let res = expand(s, &[Number(42)], &mut vars);
    assert!(res.is_ok(), "{}", res.unwrap_err());
    assert_eq!(res.unwrap(), "\\E[38;5;42m".bytes().collect::<Vec<_>>());
}

#[test]
fn test_format() {
    let mut varstruct = Variables::new();
    let vars = &mut varstruct;

    assert_eq!(
        expand(b"%p1%d%p1%.3d%p1%5d%p1%:+d", &[Number(1)], vars),
        Ok("1001    1+1".bytes().collect::<Vec<_>>())
    );
    assert_eq!(
        expand(b"%p1%o%p1%#o%p2%6.4x%p2%#6.4X", &[Number(15), Number(27)], vars),
        Ok("17017  001b0X001B".bytes().collect::<Vec<_>>())
    );
}
