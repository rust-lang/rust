// rustfmt-wrap_comments: true

/// [MyType](VeryLongPathToMyType::NoLineBreak::Here::Okay::ThatWouldBeNice::Thanks)
fn documented_with_longtype() {
    // # We're using a long type link, rustfmt should not break line
    // on the type when `wrap_comments = true`
}

/// VeryLongPathToMyType::JustMyType::But::VeryVery::Long::NoLineBreak::Here::Okay::ThatWouldBeNice::Thanks
fn documented_with_verylongtype() {
    // # We're using a long type link, rustfmt should not break line
    // on the type when `wrap_comments = true`
}
