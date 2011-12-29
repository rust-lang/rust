fn main() {
    let Π = 3.14;
    let लंच = Π * Π + 1.54;
    assert लंच - 1.54 == Π * Π;
    assert საჭმელად_გემრიელი_სადილი() == 0;
}

fn საჭმელად_გემრიელი_სადილი() -> int {

    // Lunch in several languages.

    let ランチ = 10;
    let 午餐 = 10;

    let ארוחת_צהריי = 10;
    let غداء = 10;
    let լանչ = 10;
    let обед = 10;
    let абед = 10;
    let μεσημεριανό = 10;
    let hádegismatur = 10;
    let ручек = 10;

    let ăn_trưa = 10;
    let อาหารกลางวัน = 10;

    // Lunchy arithmetic, mm.

    assert hádegismatur * ручек * обед == 1000;
    assert 10 ==  ארוחת_צהריי;
    assert ランチ + 午餐 + μεσημεριανό == 30;
    assert ăn_trưa + อาหารกลางวัน == 20;
    ret (абед + լանչ) >> غداء;
}
