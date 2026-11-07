// Non-regression test for issue #122674: a change in the format args visitor missed nested awaits.

//@ edition: 2021
//@ check-pass

pub fn f1() -> impl std::future::Future<Output = Result<(), String>> + Send {
    async {
        should_work().await?;
        Ok(())
    }
}

async fn should_work() -> Result<String, String> {
    let x = 1;
    Err(format!("test: {}: {}", x, inner().await?))
}

async fn inner() -> Result<String, String> {
    Ok("test".to_string())
}

fn main() {}
