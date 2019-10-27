import * as vscode from 'vscode'
import { TextMateRuleSettings } from './scopes'




let mappings = new Map<string, string[]>()


const defaultMapping = new Map<string, string[]>([
    ['keyword.unsafe', ['storage.modifier', 'keyword.other', 'keyword.control']],
    ['function', ['entity.name.function']],
    ['parameter', ['variable.parameter']],
    ['type', ['entity.name.type']],
    ['builtin', ['variable.language', 'support.type', 'support.type']],
    ['text', ['string', 'string.quoted', 'string.regexp']],
    ['attribute', ['keyword']],
    ['literal', ['string', 'string.quoted', 'string.regexp']],
    ['macro', ['support.other']],
    ['variable.mut', ['variable']],
    ['field', ['variable.object.property']],
    ['module', ['entity.name.section']]
]
)
function find(scope: string): string[] {
    return mappings.get(scope) || []
}

export function toRule(scope: string, intoRule: (scope: string) => TextMateRuleSettings | undefined): TextMateRuleSettings | undefined {
    return find(scope).map(intoRule).find(rule => rule !== null)
}


export function load() {
    const configuration = vscode.workspace
        .getConfiguration('rust-analyzer')
        .get('scopeMappings') as Map<string, string[]> | undefined || new Map()

    mappings = new Map([...Array.from(defaultMapping.entries()), ...Array.from(configuration.entries())]);


}