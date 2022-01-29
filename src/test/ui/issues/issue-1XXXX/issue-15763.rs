// run-pass
#![allow(unreachable_code)]
#![feature(box_syntax)]

#[derive(PartialEq, Debug)]
struct Bar {
    x: isize
}
impl Drop for Bar {
    fn drop(&mut self) {
        assert_eq!(self.x, 22);
    }
}

#[derive(PartialEq, Debug)]
struct Foo {
    x: Bar,
    a: isize
}

fn foo() -> Result<Foo, isize> {
    return Ok(Foo {
        x: Bar { x: 22 },
        a: return Err(32)
    });
}

fn baz() -> Result<Foo, isize> {
    Ok(Foo {
        x: Bar { x: 22 },
        a: return Err(32)
    })
}

// explicit immediate return
fn aa() -> isize {
    return 3;
}

// implicit immediate return
fn bb() -> isize {
    3
}

// implicit outptr return
fn cc() -> Result<isize, isize> {
    Ok(3)
}

// explicit outptr return
fn dd() -> Result<isize, isize> {
    return Ok(3);
}

trait A {
    fn aaa(&self) -> isize {
        3
    }
    fn bbb(&self) -> isize {
        return 3;
    }
    fn ccc(&self) -> Result<isize, isize> {
        Ok(3)
    }
    fn ddd(&self) -> Result<isize, isize> {
        return Ok(3);
    }
}

impl A for isize {}

fn main() {
    assert_eq!(foo(), Err(32));
    assert_eq!(baz(), Err(32));

    assert_eq!(aa(), 3);
    assert_eq!(bb(), 3);
    assert_eq!(cc().unwrap(), 3);
    assert_eq!(dd().unwrap(), 3);

    let i = box 32isize as Box<dyn A>;
    assert_eq!(i.aaa(), 3);
    let i = box 32isize as Box<dyn A>;
    assert_eq!(i.bbb(), 3);
    let i = box 32isize as Box<dyn A>;
    assert_eq!(i.ccc().unwrap(), 3);
    let i = box 32isize as Box<dyn A>;
    assert_eq!(i.ddd().unwrap(), 3);
}
