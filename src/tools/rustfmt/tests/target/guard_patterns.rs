#![feature(guard_patterns)]

fn main() {
    match user.subscription_plan() {
        (Plan::Regular if user.credit() >= 100) | (Plan::Premium if user.credit() >= 80) => {
            // Complete the transaction.
        }
        _ => {
            // The user doesn't have enough credit, return an error message.
        }
    }
}
