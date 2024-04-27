use super::ordinalize;

#[test]
fn test_ordinalize() {
    assert_eq!(ordinalize(1), "1st");
    assert_eq!(ordinalize(2), "2nd");
    assert_eq!(ordinalize(3), "3rd");
    assert_eq!(ordinalize(4), "4th");
    assert_eq!(ordinalize(5), "5th");
    // ...
    assert_eq!(ordinalize(10), "10th");
    assert_eq!(ordinalize(11), "11th");
    assert_eq!(ordinalize(12), "12th");
    assert_eq!(ordinalize(13), "13th");
    assert_eq!(ordinalize(14), "14th");
    // ...
    assert_eq!(ordinalize(20), "20th");
    assert_eq!(ordinalize(21), "21st");
    assert_eq!(ordinalize(22), "22nd");
    assert_eq!(ordinalize(23), "23rd");
    assert_eq!(ordinalize(24), "24th");
    // ...
    assert_eq!(ordinalize(30), "30th");
    assert_eq!(ordinalize(31), "31st");
    assert_eq!(ordinalize(32), "32nd");
    assert_eq!(ordinalize(33), "33rd");
    assert_eq!(ordinalize(34), "34th");
    // ...
    assert_eq!(ordinalize(7010), "7010th");
    assert_eq!(ordinalize(7011), "7011th");
    assert_eq!(ordinalize(7012), "7012th");
    assert_eq!(ordinalize(7013), "7013th");
    assert_eq!(ordinalize(7014), "7014th");
    // ...
    assert_eq!(ordinalize(7020), "7020th");
    assert_eq!(ordinalize(7021), "7021st");
    assert_eq!(ordinalize(7022), "7022nd");
    assert_eq!(ordinalize(7023), "7023rd");
    assert_eq!(ordinalize(7024), "7024th");
}
