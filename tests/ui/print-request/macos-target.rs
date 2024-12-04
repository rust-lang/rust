//@ only-apple
//@ compile-flags: --print deployment-target
//@ normalize-stdout-test: "\w*_DEPLOYMENT_TARGET" -> "$$OS_DEPLOYMENT_TARGET"
//@ normalize-stdout-test: "\d+\." -> "$$CURRENT_MAJOR_VERSION."
//@ normalize-stdout-test: "\d+" -> "$$CURRENT_MINOR_VERSION"
//@ check-pass

fn main() {}
