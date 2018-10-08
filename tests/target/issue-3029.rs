fn foo() {
    EvaluateJSReply::NumberValue(
        match FromJSValConvertible::from_jsval(cx, rval.handle(), ()) {
            Ok(ConversionResult::Success(v)) => v,
            _ => unreachable!(),
        },
    )
}

fn bar() {
    {
        {
            EvaluateJSReply::NumberValue(
                match FromJSValConvertible::from_jsval(cx, rval.handle(), ()) {
                    Ok(ConversionResult::Success(v)) => v,
                    _ => unreachable!(),
                },
            )
        }
    }
}
