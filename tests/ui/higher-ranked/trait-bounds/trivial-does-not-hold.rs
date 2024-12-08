// Minimized test from #59311.

pub fn crash()
where
    for<'a> &'a (): 'static,
{
    || {};
    //~^ ERROR higher-ranked lifetime error
}

fn main() {}
