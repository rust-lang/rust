//@ known-bug: #135039
//@ edition:2021

pub type UserId<Backend> = <<Backend as AuthnBackend>::User as AuthUser>::Id;

pub trait AuthUser {
    type Id;
}

pub trait AuthnBackend {
    type User: AuthUser;
}

pub struct AuthSession<Backend: AuthnBackend> {
    user: Option<Backend::User>,
    data: Option<UserId<Backend>>,
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
    query: Q,
) -> Result<Q::Output, ()> {
    let user = auth.user;
    query.run().await
}
