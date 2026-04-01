// rustfmt-format_code_in_doc_comments: true

struct TestStruct {
    position_currency: String, // Currency for position of this contract. If not null, 1 contract = 1 positionCurrency.
    pu: Option<i64>, // Previous event update sequense ("u" of previous message), -1 also means None
}
