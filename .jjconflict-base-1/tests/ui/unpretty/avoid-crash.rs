//@ normalize-stderr: "error `.*`" -> "$$ERROR_MESSAGE"
//@ compile-flags: -o. -Zunpretty=ast-tree

fn main() {}

//~? ERROR failed to write `.` due to error
