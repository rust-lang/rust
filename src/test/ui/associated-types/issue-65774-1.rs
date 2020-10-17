#![feature(associated_type_defaults)]

trait MyDisplay { fn method(&self) { } }

impl<'a, T: MyDisplay> MyDisplay for &'a mut T { }

struct T;

trait MPU {
    type MpuConfig: MyDisplay = T;
    //~^ ERROR the trait bound `T: MyDisplay` is not satisfied
}

struct S;

impl MPU for S { }

trait MyWrite {
    fn my_write(&self, _: &dyn MyDisplay) { }
}

trait ProcessType {
    fn process_detail_fmt(&self, _: &mut dyn MyWrite);
}

struct Process;

impl ProcessType for Process {
    fn process_detail_fmt(&self, writer: &mut dyn MyWrite)
    {

        let mut val: Option<<S as MPU>::MpuConfig> = None;
        let valref: &mut <S as MPU>::MpuConfig = val.as_mut().unwrap();

        // // This causes a different ICE (but its similar if you squint right):
        // //
        // // `Unimplemented` selecting `Binder(<T as MyDisplay>)` during codegen
        //
        // writer.my_write(valref)

        // This one causes the ICE:
        // FulfillmentError(Obligation(predicate=Binder(TraitPredicate(<T as MyDisplay>)),
        // depth=1),Unimplemented)
        let closure = |config: &mut <S as MPU>::MpuConfig| writer.my_write(&config);
        //~^ ERROR the trait bound `T: MyDisplay` is not satisfied
        closure(valref);
    }
}

fn create() -> &'static dyn ProcessType {
    let input: Option<&mut Process> = None;
    let process: &mut Process = input.unwrap();
    process
}

pub fn main() {
    create();
}
