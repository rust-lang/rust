# Debugger Internals

It is the debugger's job to convert the debug info into an in-memory representation. Both the
interpretation of the debug info and the in-memory representation are arbitrary; anything will do
so long as meaningful information can be reconstructed while the program is running. The pipeline
from raw debug info to usable types can be quite complicated.

Once the information is in a workable format, the debugger front-end then must provide a way to
interpret and display the data, a way for users to interact with it, and an API for extensibility.

Debuggers are vast systems and cannot be covered completely here. This section will provide a brief
overview of the subsystems directly relevant to the Rust debugging experience.

Microsoft's debugging engine is closed source, so it will not be covered here.