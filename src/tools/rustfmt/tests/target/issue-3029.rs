fn keep_if() {
    {
        {
            {
                EvaluateJSReply::NumberValue(
                    if FromJSValConvertible::from_jsval(cx, rval.handle(), ()) {
                        unimplemented!();
                    },
                )
            }
        }
    }
}

fn keep_if_let() {
    {
        {
            {
                EvaluateJSReply::NumberValue(
                    if let Some(e) = FromJSValConvertible::from_jsval(cx, rval.handle(), ()) {
                        unimplemented!();
                    },
                )
            }
        }
    }
}

fn keep_for() {
    {
        {
            {
                EvaluateJSReply::NumberValue(
                    for conv in FromJSValConvertible::from_jsval(cx, rval.handle(), ()) {
                        unimplemented!();
                    },
                )
            }
        }
    }
}

fn keep_loop() {
    {
        {
            {
                EvaluateJSReply::NumberValue(loop {
                    FromJSValConvertible::from_jsval(cx, rval.handle(), ());
                })
            }
        }
    }
}

fn keep_while() {
    {
        {
            {
                EvaluateJSReply::NumberValue(
                    while FromJSValConvertible::from_jsval(cx, rval.handle(), ()) {
                        unimplemented!();
                    },
                )
            }
        }
    }
}

fn keep_while_let() {
    {
        {
            {
                EvaluateJSReply::NumberValue(
                    while let Some(e) = FromJSValConvertible::from_jsval(cx, rval.handle(), ()) {
                        unimplemented!();
                    },
                )
            }
        }
    }
}

fn keep_match() {
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
