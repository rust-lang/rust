// edition:2018

async fn bar() -> Result<(), ()> {
    Ok(())
}

async fn foo1() -> Result<(), ()> {
    let _ = await bar(); //~ ERROR incorrect use of `await`
    Ok(())
}
async fn foo2() -> Result<(), ()> {
    let _ = await? bar(); //~ ERROR incorrect use of `await`
    Ok(())
}
async fn foo3() -> Result<(), ()> {
    let _ = await bar()?; //~ ERROR incorrect use of `await`
    Ok(())
}
async fn foo21() -> Result<(), ()> {
    let _ = await { bar() }; //~ ERROR incorrect use of `await`
    Ok(())
}
async fn foo22() -> Result<(), ()> {
    let _ = await(bar()); //~ ERROR incorrect use of `await`
    Ok(())
}
async fn foo23() -> Result<(), ()> {
    let _ = await { bar() }?; //~ ERROR incorrect use of `await`
    Ok(())
}
async fn foo4() -> Result<(), ()> {
    let _ = (await bar())?; //~ ERROR incorrect use of `await`
    Ok(())
}
async fn foo5() -> Result<(), ()> {
    let _ = bar().await(); //~ ERROR incorrect use of `await`
    Ok(())
}
async fn foo6() -> Result<(), ()> {
    let _ = bar().await()?; //~ ERROR incorrect use of `await`
    Ok(())
}
async fn foo7() -> Result<(), ()> {
    let _ = bar().await; // OK
    Ok(())
}
async fn foo8() -> Result<(), ()> {
    let _ = bar().await?; // OK
    Ok(())
}
fn foo9() -> Result<(), ()> {
    let _ = await bar(); //~ ERROR `await` is only allowed inside `async` functions and blocks
    //~^ ERROR incorrect use of `await`
    Ok(())
}
fn foo10() -> Result<(), ()> {
    let _ = await? bar(); //~ ERROR `await` is only allowed inside `async` functions and blocks
    //~^ ERROR incorrect use of `await`
    Ok(())
}
fn foo11() -> Result<(), ()> {
    let _ = await bar()?; //~ ERROR incorrect use of `await`
    Ok(())
}
fn foo12() -> Result<(), ()> {
    let _ = (await bar())?; //~ ERROR `await` is only allowed inside `async` functions and blocks
    //~^ ERROR incorrect use of `await`
    Ok(())
}
fn foo13() -> Result<(), ()> {
    let _ = bar().await(); //~ ERROR `await` is only allowed inside `async` functions and blocks
    //~^ ERROR incorrect use of `await`
    Ok(())
}
fn foo14() -> Result<(), ()> {
    let _ = bar().await()?; //~ ERROR `await` is only allowed inside `async` functions and blocks
    //~^ ERROR incorrect use of `await`
    Ok(())
}
fn foo15() -> Result<(), ()> {
    let _ = bar().await; //~ ERROR `await` is only allowed inside `async` functions and blocks
    Ok(())
}
fn foo16() -> Result<(), ()> {
    let _ = bar().await?; //~ ERROR `await` is only allowed inside `async` functions and blocks
    Ok(())
}
fn foo24() -> Result<(), ()> {
    fn foo() -> Result<(), ()> {
        let _ = bar().await?; //~ ERROR `await` is only allowed inside `async` functions and blocks
        Ok(())
    }
    foo()
}
fn foo25() -> Result<(), ()> {
    let foo = || {
        let _ = bar().await?; //~ ERROR `await` is only allowed inside `async` functions and blocks
        Ok(())
    };
    foo()
}

async fn foo26() -> Result<(), ()> {
    let _ = await!(bar()); //~ ERROR incorrect use of `await`
    Ok(())
}
async fn foo27() -> Result<(), ()> {
    let _ = await!(bar())?; //~ ERROR incorrect use of `await`
    Ok(())
}
fn foo28() -> Result<(), ()> {
    fn foo() -> Result<(), ()> {
        let _ = await!(bar())?; //~ ERROR incorrect use of `await`
        //~^ ERROR `await` is only allowed inside `async` functions
        Ok(())
    }
    foo()
}
fn foo29() -> Result<(), ()> {
    let foo = || {
        let _ = await!(bar())?; //~ ERROR incorrect use of `await`
        //~^ ERROR `await` is only allowed inside `async` functions
        Ok(())
    };
    foo()
}

fn main() {
    match await { await => () }
    //~^ ERROR expected expression, found `=>`
    //~| ERROR incorrect use of `await`
} //~ ERROR expected one of `.`, `?`, `{`, or an operator, found `}`
