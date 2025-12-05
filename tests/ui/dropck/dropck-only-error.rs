// Test that we don't ICE for a typeck error that only shows up in dropck
// issue #135039

pub trait AuthUser {
    type Id;
}

pub trait AuthnBackend {
    type User: AuthUser;
}

pub struct AuthSession<Backend: AuthnBackend> {
    data: Option<<<Backend as AuthnBackend>::User as AuthUser>::Id>,
}

pub trait Authz: Sized {
    type AuthnBackend: AuthnBackend<User = Self>;
}

pub fn run_query<User: Authz>(auth: AuthSession<User::AuthnBackend>) {}
//~^ ERROR the trait bound `User: AuthUser` is not satisfied [E0277]

fn main() {}
