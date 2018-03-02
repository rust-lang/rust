// Preserve two trailing whitespaces in doc comment,
// but trim any whitespaces in normal comment.

//! hello world  
//! hello world 

/// hello world    
/// hello world 
/// hello world  
fn foo() {
    // hello world  
    // hello world 
    let x = 3;
    println!("x = {}", x);
}
