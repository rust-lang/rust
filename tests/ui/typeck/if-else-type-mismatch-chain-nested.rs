fn main() {
    let oa = Some(1);
    let oa2 = Some(1);
    let _v = if let Some(a) = oa {
        Some(&a)
    } else if let Some(a) = oa2 {
        let _f = 1;

        let _h = if true {
            1
        } else if true {
            "2" //~ ERROR `if` and `else` have incompatible types [E0308]
        } else {
            3
        };

        &Some(a) //~ ERROR `if` and `else` have incompatible types [E0308]
    } else {
        None
    };
}
