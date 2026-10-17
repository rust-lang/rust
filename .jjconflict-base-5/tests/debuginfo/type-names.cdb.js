// Helper functions for running the type-names.rs test under CDB

// CDB exposes an "object model" via JavaScript that allows you to inspect debugging info - in this
// case we want to ask the object model for the return and parameter types for a local variable
// that is a function pointer to make sure that we are emitting the function pointer type in such a
// way that CDB understands how to parse it.

"use strict";

function getFunctionDetails(name)
{
    var localVariable = host.currentThread.Stack.Frames[0].LocalVariables[name];
    var functionPointerType = localVariable.targetType.genericArguments[0];
    var functionType = functionPointerType.baseType;
    host.diagnostics.debugLog("Return Type: ", functionType.functionReturnType, "\n");
    host.diagnostics.debugLog("Parameter Types: ", functionType.functionParameterTypes, "\n");
}
