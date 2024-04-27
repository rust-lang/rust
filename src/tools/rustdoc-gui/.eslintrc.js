module.exports = {
    "env": {
        "browser": true,
        "node": true,
        "es6": true
    },
    "extends": "eslint:recommended",
    "parserOptions": {
        "ecmaVersion": 2018,
        "sourceType": "module"
    },
    "rules": {
        "linebreak-style": [
            "error",
            "unix"
        ],
        "semi": [
            "error",
            "always"
        ],
        "quotes": [
            "error",
            "double"
        ],
        "linebreak-style": [
            "error",
            "unix"
        ],
        "no-trailing-spaces": "error",
        "no-var": ["error"],
        "prefer-const": ["error"],
        "prefer-arrow-callback": ["error"],
        "brace-style": [
            "error",
            "1tbs",
            { "allowSingleLine": false }
        ],
        "keyword-spacing": [
            "error",
            { "before": true, "after": true }
        ],
        "arrow-spacing": [
            "error",
            { "before": true, "after": true }
        ],
        "key-spacing": [
            "error",
            { "beforeColon": false, "afterColon": true, "mode": "strict" }
        ],
        "func-call-spacing": ["error", "never"],
        "space-infix-ops": "error",
        "space-before-function-paren": ["error", "never"],
        "space-before-blocks": "error",
        "comma-dangle": ["error", "always-multiline"],
        "comma-style": ["error", "last"],
        "max-len": ["error", { "code": 100, "tabWidth": 4 }],
        "eol-last": ["error", "always"],
        "arrow-parens": ["error", "as-needed"],
        "no-unused-vars": [
            "error",
            {
                "argsIgnorePattern": "^_",
                "varsIgnorePattern": "^_"
            }
        ],
        "eqeqeq": "error",
        "no-const-assign": "error",
        "no-debugger": "error",
        "no-dupe-args": "error",
        "no-dupe-else-if": "error",
        "no-dupe-keys": "error",
        "no-duplicate-case": "error",
        "no-ex-assign": "error",
        "no-fallthrough": "error",
        "no-invalid-regexp": "error",
        "no-import-assign": "error",
        "no-self-compare": "error",
        "no-template-curly-in-string": "error",
        "block-scoped-var": "error",
        "guard-for-in": "error",
        "no-alert": "error",
        "no-confusing-arrow": "error",
        "no-div-regex": "error",
        "no-floating-decimal": "error",
        "no-implicit-globals": "error",
        "no-implied-eval": "error",
        "no-label-var": "error",
        "no-lonely-if": "error",
        "no-mixed-operators": "error",
        "no-multi-assign": "error",
        "no-return-assign": "error",
        "no-script-url": "error",
        "no-sequences": "error",
        "no-div-regex": "error",
    }
};
