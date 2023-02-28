static CONST: Option<dyn Fn(& _)> = None;
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for static items [E0121]
