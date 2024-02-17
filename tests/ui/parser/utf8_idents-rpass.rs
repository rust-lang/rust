//@ run-pass
//
#![allow(non_snake_case)]

pub fn main() {
    let ε = 0.00001f64;
    let Π = 3.14f64;
    let लंच = Π * Π + 1.54;
    assert!(((लंच - 1.54) - (Π * Π)).abs() < ε);
    assert_eq!(საჭმელად_გემრიელი_სადილი(), 0);
}

fn საჭმელად_გემრიელი_სადილი() -> isize {

    // Lunch in several languages.

    let ランチ = 10;
    let 午餐 = 10;

    let ארוחת_צהריי = 10;
    let غداء = 10_usize;
    let լանչ = 10;
    let обед = 10;
    let абед = 10;
    let μεσημεριανό = 10;
    let hádegismatur = 10;
    let ручек = 10;

    let ăn_trưa = 10;
    let อาหารกลางวัน = 10;

    // Lunchy arithmetic, mm.

    assert_eq!(hádegismatur * ручек * обед, 1000);
    assert_eq!(10, ארוחת_צהריי);
    assert_eq!(ランチ + 午餐 + μεσημεριανό, 30);
    assert_eq!(ăn_trưa + อาหารกลางวัน, 20);
    return (абед + լանչ) >> غداء;
}
