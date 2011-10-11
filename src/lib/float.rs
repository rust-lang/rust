fn float_to_str(num: float, digits: uint) -> str {
    let accum = if num < 0.0 { num = -num; "-" } else { "" };
    let trunc = num as uint;
    let frac = num - (trunc as float);
    accum += uint::str(trunc);
    if frac == 0.0 || digits == 0u { ret accum; }
    accum += ".";
    while digits > 0u && frac > 0.0 {
        frac *= 10.0;
        let digit = frac as uint;
        accum += uint::str(digit);
        frac -= digit as float;
        digits -= 1u;
    }
    ret accum;
}

fn str_to_float(num: str) -> float {
    let digits = str::split(num, '.' as u8);
    let total = int::from_str(digits[0]) as float;

    fn dec_val(c: char) -> int { ret (c as int) - ('0' as int); }

    let right = digits[1];
    let len = str::char_len(digits[1]);
    let i = 1u;
    while (i < len) {
        total += dec_val(str::pop_char(right)) as float /
                 (int::pow(10, i) as float);
        i += 1u;
    }
    ret total;
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
