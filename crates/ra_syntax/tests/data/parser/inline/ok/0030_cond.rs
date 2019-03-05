fn foo() { if let Some(_) = None {} }
fn bar() {
    if let Some(_) | Some(_) = None {}
    if let | Some(_) = None {}
    while let Some(_) | Some(_) = None {}
    while let | Some(_) = None {}
}
