//@ only-apple
//@ compile-flags: --print deployment-target
//@ normalize-stdout: "\w*_DEPLOYMENT_TARGET" -> "$$OS_DEPLOYMENT_TARGET"
//@ normalize-stdout: "\d+\." -> "$$CURRENT_MAJOR_VERSION."
//@ normalize-stdout: "\d+" -> "$$CURRENT_MINOR_VERSION"
//@ check-pass

fn main() {}
