enum Status {
    Active,
    Inactive,
    Pending,
}
fn main() {
    let message: &str; // Declared but not initialized
    let current_status = Status::Pending;
    match current_status {
        Status::Active => {
            message = "System is live.";
        }
        Status::Inactive => {
            message = "System is down.";
        }
        Status::Pending => {
            println!("{message}"); //~ ERROR E0381
        }
    }
}
