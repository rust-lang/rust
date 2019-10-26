import * as fs from 'fs'
import * as path from 'path'
import * as vscode from 'vscode'



export interface TextMateRule {
    scope: string | string[]
    settings: TextMateRuleSettings
}

export interface TextMateRuleSettings {
    foreground: string | undefined
    background: string | undefined
    fontStyle: string | undefined
}

// Current theme colors
const colors = new Map<string, TextMateRuleSettings>()

export function find(scope: string): TextMateRuleSettings | undefined {
    return colors.get(scope)
}

// Load all textmate scopes in the currently active theme
export function load() {
    // Remove any previous theme
    colors.clear()
    // Find out current color theme
    const themeName = vscode.workspace.getConfiguration('workbench').get('colorTheme')

    if (typeof themeName !== 'string') {
        // console.warn('workbench.colorTheme is', themeName)
        return
    }
    // Try to load colors from that theme
    try {
        loadThemeNamed(themeName)
    } catch (e) {
        // console.warn('failed to load theme', themeName, e)
    }
}

// Find current theme on disk
function loadThemeNamed(themeName: string) {

    const themePaths = vscode.extensions.all
        .filter(extension => extension.extensionKind === vscode.ExtensionKind.UI)
        .filter(extension => extension.packageJSON.contributes)
        .filter(extension => extension.packageJSON.contributes.themes)
        .reduce((list, extension) => {
            const paths = extension.packageJSON.contributes.themes
                .filter((element: any) => (element.id || element.label) === themeName)
                .map((element: any) => path.join(extension.extensionPath, element.path))
            return list.concat(paths)
        }, Array<string>());


    themePaths.forEach(loadThemeFile);

    const customization: any = vscode.workspace.getConfiguration('editor').get('tokenColorCustomizations');
    if (customization && customization.textMateRules) {
        loadColors(customization.textMateRules)
    }
}

function loadThemeFile(themePath: string) {

    if (checkFileExists(themePath)) {
        const themeContentText: string = readFileText(themePath)

        const themeContent: any = JSON.parse(themeContentText)

        if (themeContent && themeContent.tokenColors) {
            loadColors(themeContent.tokenColors)
            if (themeContent.include) {
                // parse included theme file
                const includedThemePath: string = path.join(path.dirname(themePath), themeContent.include)
                loadThemeFile(includedThemePath)
            }
        }
    }
}
function mergeRuleSettings(defaultSetting: TextMateRuleSettings, override: TextMateRuleSettings): TextMateRuleSettings {
    const mergedRule = defaultSetting;

    mergedRule.background = override.background || defaultSetting.background
    mergedRule.foreground = override.foreground || defaultSetting.foreground
    mergedRule.fontStyle = override.fontStyle || defaultSetting.foreground;

    return mergedRule;
}

function loadColors(textMateRules: TextMateRule[]): void {
    for (const rule of textMateRules) {

        if (typeof rule.scope === 'string') {
            const existingRule = colors.get(rule.scope);
            if (existingRule) {
                colors.set(rule.scope, mergeRuleSettings(existingRule, rule.settings))
            }
            else {
                colors.set(rule.scope, rule.settings)
            }
        } else if (rule.scope instanceof Array) {
            for (const scope of rule.scope) {
                const existingRule = colors.get(scope);
                if (existingRule) {
                    colors.set(scope, mergeRuleSettings(existingRule, rule.settings))
                }
                else {
                    colors.set(scope, rule.settings)
                }
            }
        }
    }
}

function checkFileExists(filePath: string): boolean {

    const stats = fs.statSync(filePath);
    if (stats && stats.isFile()) {
        return true;
    } else {
        // console.warn('no such file', filePath)
        return false;
    }


}

function readFileText(filePath: string, encoding: string = 'utf8'): string {
    return fs.readFileSync(filePath, encoding);
}