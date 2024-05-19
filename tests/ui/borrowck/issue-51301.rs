use std::any::TypeId;
use std::collections::HashMap;
use std::hash::Hash;

trait State {
    type EventType;
    fn get_type_id_of_state(&self) -> TypeId;
}

struct StateMachine<EventType: Hash + Eq> {
    current_state: Box<dyn State<EventType = EventType>>,
    transition_table:
        HashMap<TypeId, HashMap<EventType, fn() -> Box<dyn State<EventType = EventType>>>>,
}

impl<EventType: Hash + Eq> StateMachine<EventType> {
    fn inner_process_event(&mut self, event: EventType) -> Result<(), i8> {
        let new_state_creation_function = self
            .transition_table
            .iter()
            .find(|(&event_typeid, _)| event_typeid == self.current_state.get_type_id_of_state())
            .ok_or(1)?
            .1
            .iter()
            .find(|(&event_type, _)| event == event_type)
            //~^ ERROR cannot move out of a shared reference
            .ok_or(2)?
            .1;

        self.current_state = new_state_creation_function();
        Ok(())
    }
}

fn main() {}
