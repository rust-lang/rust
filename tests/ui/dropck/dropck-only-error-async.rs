// Test that we don't ICE for a typeck error that only shows up in dropck
// issue #135039
//@ edition:2018

pub trait AuthUser {
    type Id;
}

pub trait AuthnBackend {
    type User: AuthUser;
}

pub struct AuthSession<Backend: AuthnBackend> {
    user: Option<Backend::User>,
    data: Option<<<Backend as AuthnBackend>::User as AuthUser>::Id>,
}

pub trait Authz: Sized {
    type AuthnBackend: AuthnBackend<User = Self>;
}

pub trait Query<User: Authz> {
    type Output;
    async fn run(&self) -> Result<Self::Output, ()>;
}

pub async fn run_query<User: Authz, Q: Query<User> + 'static>(
    auth: AuthSession<User::AuthnBackend>,
    //~^ ERROR the trait bound `User: AuthUser` is not satisfied [E0277]
    //~| ERROR the trait bound `User: AuthUser` is not satisfied [E0277]
    query: Q,
) -> Result<Q::Output, ()> {
    let user = auth.user;
    query.run().await
}

fn main() {}
