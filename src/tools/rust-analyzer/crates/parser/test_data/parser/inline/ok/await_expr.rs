fn foo() {
    x.await;
    x.0.await;
    x.0().await?.hello();
    x.0.0.await;
    x.0. await;
}
